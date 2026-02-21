
# Get the HTML and parse it
html = await get_html(url="https://platform.claude.com/docs/en/about-claude/pricing")
soup = await beautiful_soup(html=html)

# Find all tables in the page
tables = await find_all(tag=soup, name="table")

# Extract information from each table
all_models = []

for table in tables:
    # Get all rows
    rows = await find_all(tag=table, name="tr")
    
    if len(rows) < 2:
        continue
    
    # Get headers from first row
    header_row = rows[0]
    headers = await find_all(tag=header_row, name="th")
    header_texts = []
    for h in headers:
        text = await get_text(tag=h, strip=True)
        header_texts.append(text)
    
    # Process data rows
    for i in range(1, len(rows)):
        row = rows[i]
        cells = await find_all(tag=row, name="td")
        
        if len(cells) < 3:
            continue
        
        row_data = {}
        for j in range(len(cells)):
            if j < len(header_texts):
                cell_text = await get_text(tag=cells[j], strip=True)
                row_data[header_texts[j]] = cell_text
        
        all_models.append(row_data)

# Function to extract price from string like "$5 / MTok"
def extract_price(price_str):
    if not price_str:
        return None
    # Remove "$", "/", "MTok" and whitespace
    cleaned = price_str.replace("$", "").replace("/ MTok", "").replace("/MTok", "").strip()
    try:
        return float(cleaned)
    except:
        return None

# Process each model and record info
results = []
for model_data in all_models:
    model_name = model_data.get("Model", "")
    
    # Skip deprecated models
    if "deprecated" in model_name.lower():
        continue
    
    # Get base input and output prices
    base_input_str = model_data.get("Base Input Tokens", "")
    output_str = model_data.get("Output Tokens", "")
    
    input_price = extract_price(base_input_str)
    output_price = extract_price(output_str)
    
    # Skip if we don't have valid prices
    if input_price is None or output_price is None:
        continue
    
    # Create a unique ID based on model name (lowercase, replace spaces with dashes)
    unique_id = model_name.lower().replace(" ", "-").replace(".", "-")
    
    # Extract other attributes
    attributes = {}
    for key, value in model_data.items():
        if key not in ["Model", "Base Input Tokens", "Output Tokens"]:
            attributes[key] = value
    
    # Record the model
    result = await record_model_info(
        model_information={
            "unique_id": unique_id,
            "name": model_name,
            "description": None,
            "input_mtok": input_price,
            "output_mtok": output_price,
            "attributes": attributes if attributes else None
        }
    )
    results.append(result)
