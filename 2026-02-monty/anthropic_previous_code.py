html = await get_html(url="https://platform.claude.com/docs/en/about-claude/pricing")
soup = await beautiful_soup(html=html)

# Find all tables
tables = await find_all(tag=soup, name="table")

# Process each table to find pricing information
models_data = []

for table in tables:
    rows = await find_all(tag=table, name="tr")
    
    if not rows:
        continue
    
    # Get headers
    header_row = rows[0]
    headers = await find_all(tag=header_row, name="th")
    header_texts = []
    for h in headers:
        text = await get_text(tag=h, strip=True)
        header_texts.append(text)
    
    # Check if this table has Model column
    if not header_texts or "Model" not in header_texts:
        continue
    
    # Process data rows
    for i in range(1, len(rows)):
        row = rows[i]
        cells = await find_all(tag=row, name="td")
        
        if not cells:
            continue
        
        row_data = {}
        for j in range(len(cells)):
            cell = cells[j]
            text = await get_text(tag=cell, strip=True)
            if j < len(header_texts):
                row_data[header_texts[j]] = text
        
        if row_data and "Model" in row_data:
            models_data.append(row_data)

# Helper function to extract numeric price
def extract_price(price_str):
    if not price_str or price_str == "":
        return None
    cleaned = price_str.replace("$", "").replace("/", "").replace("MTok", "").replace(" ", "")
    try:
        return float(cleaned)
    except:
        return None

# Record each model
for model in models_data:
    model_name = model.get("Model", "")
    
    if not model_name:
        continue
    
    unique_id = model_name.lower().replace(" ", "-").replace("(", "").replace(")", "")
    
    # Extract prices from various column names
    input_price = None
    output_price = None
    
    if "Base Input Tokens" in model:
        input_price = extract_price(model.get("Base Input Tokens", ""))
    elif "Input" in model:
        input_price = extract_price(model.get("Input", ""))
    elif "Batch input" in model:
        input_price = extract_price(model.get("Batch input", ""))
    
    if "Output Tokens" in model:
        output_price = extract_price(model.get("Output Tokens", ""))
    elif "Output" in model:
        output_price = extract_price(model.get("Output", ""))
    elif "Batch output" in model:
        output_price = extract_price(model.get("Batch output", ""))
    
    if input_price is None or output_price is None:
        continue
    
    # Build attributes dict
    attributes = {}
    for key, value in model.items():
        if key != "Model" and key != "Base Input Tokens" and key != "Output Tokens" and key != "Input" and key != "Output" and key != "Batch input" and key != "Batch output":
            attributes[key] = value
    
    info = f"""unique_id: {unique_id}
name: {model_name}
input_mtok: {input_price}
output_mtok: {output_price}"""
    
    if attributes:
        info += f"\nattributes: {attributes}"
    
    await record_model_info(model_information=info)