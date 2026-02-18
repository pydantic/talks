html = await get_html(url="https://platform.claude.com/docs/en/about-claude/pricing")
soup = await beautiful_soup(html=html)

tables = await find_all(tag=soup, name="table")

def parse_price(price_str):
    price_str = price_str.replace("$", "").replace("/ MTok", "").replace("/MTok", "").strip()
    if ":" in price_str:
        price_str = price_str.split(":")[1].strip()
        price_str = price_str.replace("$", "").replace("/ MTok", "").replace("/MTok", "").strip()
    try:
        return float(price_str)
    except:
        return None

models_to_record = []

# Table 0: Main pricing with prompt caching
if len(tables) > 0:
    table0 = tables[0]
    tbody0 = await find(tag=table0, name="tbody")
    if tbody0:
        rows = await find_all(tag=tbody0, name="tr")
        for row in rows:
            cells = await find_all(tag=row, name="td")
            if len(cells) >= 6:
                model_name = await get_text(tag=cells[0], strip=True)
                base_input = await get_text(tag=cells[1], strip=True)
                output = await get_text(tag=cells[5], strip=True)
                
                input_price = parse_price(base_input)
                output_price = parse_price(output)
                
                if input_price and output_price and model_name:
                    unique_id = model_name.lower().replace(" ", "-").replace("(", "").replace(")", "")
                    models_to_record.append({
                        "unique_id": unique_id,
                        "name": model_name,
                        "description": "Standard pricing",
                        "input_mtok": input_price,
                        "output_mtok": output_price,
                        "attributes": {"pricing_type": "standard"}
                    })

# Table 1: Claude Opus 4 extended context
if len(tables) > 1:
    table1 = tables[1]
    tbody1 = await find(tag=table1, name="tbody")
    if tbody1:
        rows = await find_all(tag=tbody1, name="tr")
        for row in rows:
            cells = await find_all(tag=row, name="td")
            if len(cells) >= 3:
                context = await get_text(tag=cells[0], strip=True)
                input_price_str = await get_text(tag=cells[1], strip=True)
                output_price_str = await get_text(tag=cells[2], strip=True)
                
                input_price = parse_price(input_price_str)
                output_price = parse_price(output_price_str)
                
                if input_price and output_price:
                    context_clean = context.replace(" ", "-").replace("≤", "lte").replace(">", "gt")
                    unique_id = f"claude-opus-4-extended-{context_clean}"
                    models_to_record.append({
                        "unique_id": unique_id,
                        "name": f"Claude Opus 4 Extended Context ({context})",
                        "description": f"Extended context pricing for {context}",
                        "input_mtok": input_price,
                        "output_mtok": output_price,
                        "attributes": {"pricing_type": "extended_context", "context_window": context}
                    })

# Table 2: Batch API pricing
if len(tables) > 2:
    table2 = tables[2]
    tbody2 = await find(tag=table2, name="tbody")
    if tbody2:
        rows = await find_all(tag=tbody2, name="tr")
        for row in rows:
            cells = await find_all(tag=row, name="td")
            if len(cells) >= 3:
                model_name = await get_text(tag=cells[0], strip=True)
                batch_input = await get_text(tag=cells[1], strip=True)
                batch_output = await get_text(tag=cells[2], strip=True)
                
                input_price = parse_price(batch_input)
                output_price = parse_price(batch_output)
                
                if input_price and output_price and model_name:
                    unique_id = model_name.lower().replace(" ", "-").replace("(", "").replace(")", "") + "-batch"
                    models_to_record.append({
                        "unique_id": unique_id,
                        "name": f"{model_name} (Batch API)",
                        "description": "Batch API pricing",
                        "input_mtok": input_price,
                        "output_mtok": output_price,
                        "attributes": {"pricing_type": "batch_api"}
                    })

# Table 3: Extended context for multiple models
if len(tables) > 3:
    table3 = tables[3]
    tbody3 = await find(tag=table3, name="tbody")
    if tbody3:
        rows = await find_all(tag=tbody3, name="tr")
        i = 0
        while i < len(rows):
            row = rows[i]
            cells = await find_all(tag=row, name="td")
            
            if len(cells) >= 3:
                model_cell_text = await get_text(tag=cells[0], strip=True)
                
                if model_cell_text and "Input:" in await get_text(tag=cells[1], strip=True):
                    model_name = model_cell_text
                    input_lte200k = await get_text(tag=cells[1], strip=True)
                    input_gt200k = await get_text(tag=cells[2], strip=True)
                    
                    if i + 1 < len(rows):
                        next_row = rows[i + 1]
                        next_cells = await find_all(tag=next_row, name="td")
                        if len(next_cells) >= 3:
                            output_lte200k = await get_text(tag=next_cells[1], strip=True)
                            output_gt200k = await get_text(tag=next_cells[2], strip=True)
                            
                            input_price_lte = parse_price(input_lte200k)
                            output_price_lte = parse_price(output_lte200k)
                            input_price_gt = parse_price(input_gt200k)
                            output_price_gt = parse_price(output_gt200k)
                            
                            if input_price_lte and output_price_lte:
                                unique_id = model_name.lower().replace(" ", "-").replace("/", "-").replace("(", "").replace(")", "") + "-extended-lte200k"
                                models_to_record.append({
                                    "unique_id": unique_id,
                                    "name": f"{model_name} Extended (≤ 200K tokens)",
                                    "description": f"Extended context pricing for {model_name} with ≤ 200K input tokens",
                                    "input_mtok": input_price_lte,
                                    "output_mtok": output_price_lte,
                                    "attributes": {"pricing_type": "extended_context", "context_window": "≤ 200K input tokens"}
                                })
                            
                            if input_price_gt and output_price_gt:
                                unique_id = model_name.lower().replace(" ", "-").replace("/", "-").replace("(", "").replace(")", "") + "-extended-gt200k"
                                models_to_record.append({
                                    "unique_id": unique_id,
                                    "name": f"{model_name} Extended (> 200K tokens)",
                                    "description": f"Extended context pricing for {model_name} with > 200K input tokens",
                                    "input_mtok": input_price_gt,
                                    "output_mtok": output_price_gt,
                                    "attributes": {"pricing_type": "extended_context", "context_window": "> 200K input tokens"}
                                })
                            
                            i += 2
                            continue
            i += 1

for model in models_to_record:
    await record_model_info(model_information=model)