
html = await get_html(url="https://groq.com/pricing")
tag = await beautiful_soup(html=html)
tables = await select(tag=tag, selector="table")

async def extract_table_rows(table):
    rows = await select(tag=table, selector="tbody tr")
    result = []
    for row in rows:
        cells = await select(tag=row, selector="td")
        cell_texts = []
        for cell in cells:
            main_spans = await select(tag=cell, selector=".PricingTable_pricingTable-table__contents-inner__IXD5F > span")
            if main_spans:
                text = await get_text(tag=main_spans[0], strip=True)
                cell_texts.append(text)
            else:
                text = await get_text(tag=cell, strip=True)
                cell_texts.append(text[:80])
        # Extract model ID from docs link
        links = await select(tag=row, selector="a")
        model_id = None
        for link in links:
            href = await get(tag=link, key="href")
            if href and isinstance(href, str) and "/docs/model/" in href:
                model_id = href.split("/docs/model/")[-1]
                break
        cell_texts.append(model_id)
        result.append(cell_texts)
    return result

# Table 0: LLM models [name, speed_tps, input_price, output_price, cta, model_id]
# Table 1: TTS models [name, chars_per_s, price_per_m_chars, cta, model_id]
# Table 2: ASR models [name, speed_factor, price_per_hour, cta, model_id]
llm_rows = await extract_table_rows(tables[0])
tts_rows = await extract_table_rows(tables[1])
asr_rows = await extract_table_rows(tables[2])

import asyncio

calls = []

for row in llm_rows:
    name, speed, input_price, output_price, _, model_id = row[0], row[1], row[2], row[3], row[4], row[5]
    input_mtok = float(input_price.replace("$", ""))
    output_mtok = float(output_price.replace("$", ""))
    calls.append(record_model_info(model_information={
        "unique_id": model_id or name,
        "name": name,
        "description": f"Large Language Model on Groq. Speed: {speed}",
        "input_mtok": input_mtok,
        "output_mtok": output_mtok,
        "attributes": {"speed_tps": speed, "type": "LLM"}
    }))

for row in tts_rows:
    name, chars_per_s, price_str, _, model_id = row[0], row[1], row[2], row[3], row[4]
    price = float(price_str.replace("$", ""))
    calls.append(record_model_info(model_information={
        "unique_id": model_id or name,
        "name": name,
        "description": f"Text-to-Speech model on Groq. {chars_per_s} characters/second.",
        "input_mtok": 0,
        "output_mtok": 0,
        "attributes": {"price_per_m_chars": price, "chars_per_s": int(chars_per_s), "type": "TTS"}
    }))

for row in asr_rows:
    name, speed_factor, price_str, _, model_id = row[0], row[1], row[2], row[3], row[4]
    price = float(price_str.replace("$", "").replace("*", ""))
    calls.append(record_model_info(model_information={
        "unique_id": model_id or name,
        "name": name,
        "description": f"ASR model on Groq. Speed factor: {speed_factor}. Priced per hour transcribed.",
        "input_mtok": 0,
        "output_mtok": 0,
        "attributes": {"price_per_hour_transcribed": price, "speed_factor": speed_factor, "type": "ASR"}
    }))

results = await asyncio.gather(*calls)
results
