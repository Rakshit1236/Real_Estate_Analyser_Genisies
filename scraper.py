import requests
from bs4 import BeautifulSoup
import csv
import re
import time
import math
import pandas as pd

# Configuration
BASE_URL_SALE = "https://www.magicbricks.com/property-for-sale/residential-real-estate?bedroom=&proptype=Multistorey-Apartment,Builder-Floor-Apartment,Penthouse,Studio-Apartment,Residential-House,Villa&cityName=Surat"
BASE_URL_RENT = "https://www.magicbricks.com/property-for-rent/residential-real-estate?bedroom=&proptype=Multistorey-Apartment,Builder-Floor-Apartment,Penthouse,Studio-Apartment,Residential-House,Villa&cityName=Surat"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
OUTPUT_FILE = "real_estate_data.csv"
MAX_PAGES = 5  # Limit to 5 pages for demonstration

# Financial assumptions
BANK_RATES = {
    "SBI": 8.4 / 100,
    "HDFC": 8.6 / 100,
    "ICICI": 8.7 / 100,
    "AXIS": 8.5 / 100,
}
AVG_INTEREST_RATE = sum(BANK_RATES.values()) / len(BANK_RATES)
LOAN_TENURE_YEARS = 20
DOWN_PAYMENT_PCT = 0.20  # 20% down payment
APPRECIATION_RATE = 0.05  # 5% annual property appreciation
RENT_YIELD = 0.03         # 3% of property value per year as rent (for comparison)
RENT_ESCALATION = 0.05    # 5% annual rent increase
INVEST_RETURN_RATE = 0.08 # 8% annual return on investments (FD/MF)
TAX_RATE = 0.30           # 30% income tax slab
SECTION24_LIMIT = 200000  # Interest deduction limit per year
SECTION80C_LIMIT = 150000 # Principal deduction limit per year

def clean_text(text):
    if text:
        return text.strip()
    return ""

def parse_price_to_inr(price_str):
    """Convert price strings like '₹ 1.6 Cr', '₹ 75 Lac', '₹ 25,000' to numeric INR.
    Returns float or None if parsing fails.
    """
    if not price_str or not isinstance(price_str, str):
        return None

    s = price_str
    # Remove currency symbol and commas and phrases like 'per sqft', '/month'
    s = s.replace('₹', '').replace(',', '').strip()
    s = re.sub(r"/.*$", "", s)  # remove everything after '/'

    # Handle crore/lakh/lac
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(cr|crore|lac|lakh)?", s, re.IGNORECASE)
    if not match:
        return None

    value = float(match.group(1))
    unit = match.group(2).lower() if match.group(2) else None

    if unit in ["cr", "crore"]:
        return value * 1e7  # 1 crore = 10,000,000
    if unit in ["lac", "lakh"]:
        return value * 1e5  # 1 lakh = 100,000

    # If no unit, treat as plain rupees
    return value

def calculate_emi_and_schedule(loan_amount, annual_rate, tenure_years):
    """Return EMI, total interest, total principal, yearly interest/principal lists."""
    if loan_amount <= 0:
        return 0.0, 0.0, 0.0, [], []

    r_monthly = annual_rate / 12.0
    n_months = tenure_years * 12

    # EMI formula
    factor = (1 + r_monthly) ** n_months
    emi = loan_amount * r_monthly * factor / (factor - 1)

    balance = loan_amount
    total_interest = 0.0
    total_principal = 0.0
    yearly_interest = [0.0 for _ in range(tenure_years)]
    yearly_principal = [0.0 for _ in range(tenure_years)]

    for m in range(1, n_months + 1):
        interest_m = balance * r_monthly
        principal_m = emi - interest_m
        balance -= principal_m
        total_interest += interest_m
        total_principal += principal_m

        year_idx = (m - 1) // 12
        if year_idx < tenure_years:
            yearly_interest[year_idx] += interest_m
            yearly_principal[year_idx] += principal_m

    return emi, total_interest, total_principal, yearly_interest, yearly_principal

def future_value_lumpsum(pv, annual_rate, years):
    return pv * ((1 + annual_rate) ** years)

def future_value_sip(contributions, annual_rate, years):
    """Compute future value of varying monthly SIP contributions.

    contributions: list of monthly amounts over tenure_years * 12
    """
    if not contributions:
        return 0.0

    r_monthly = annual_rate / 12.0
    fv = 0.0
    for c in contributions:
        fv = fv * (1 + r_monthly) + c
    return fv

def compute_financials_row(row):
    """Given a row with Price and For Sale/Rent, compute buy vs rent metrics.

    We primarily compute detailed metrics for 'Sale' properties where a price is available.
    """
    price_num = parse_price_to_inr(row.get('Price'))
    listing_type = row.get('For Sale/Rent', '')

    result = {
        'Numeric Price (INR)': price_num,
        'Avg Interest Rate (annual)': AVG_INTEREST_RATE,
        'Loan Tenure (years)': LOAN_TENURE_YEARS,
        'Down Payment (INR)': None,
        'Loan Amount (INR)': None,
        'Monthly EMI (INR)': None,
        'Total Interest Paid (INR)': None,
        'Total Principal Repaid (INR)': None,
        'Total EMI Paid 20y (INR)': None,
        'Total Tax Savings 20y (INR)': None,
        'Net EMI Paid 20y (INR)': None,
        'Avg Net EMI per Month (INR)': None,
        'Final Asset Value 20y (INR)': None,
        'Initial Monthly Rent (INR)': None,
        'Total Rent Paid 20y (INR)': None,
        'Lumpsum FV from Down Payment (INR)': None,
        'SIP FV from Monthly Savings (INR)': None,
        'Final Wealth - Buy (INR)': None,
        'Final Wealth - Rent (INR)': None,
        'Recommendation (Buy/Rent)': 'Data Insufficient',
    }

    # Only compute detailed metrics when we have a valid numeric price and this is a Sale listing
    if price_num is None or listing_type.lower() != 'sale':
        return result

    down_payment = price_num * DOWN_PAYMENT_PCT
    loan_amount = price_num - down_payment

    emi, total_interest, total_principal, yearly_interest, yearly_principal = calculate_emi_and_schedule(
        loan_amount, AVG_INTEREST_RATE, LOAN_TENURE_YEARS
    )

    n_months = LOAN_TENURE_YEARS * 12
    total_emi_paid = emi * n_months

    # Tax savings calculation over 20 years (Section 24(b) + 80C limits)
    total_tax_savings = 0.0
    for y in range(LOAN_TENURE_YEARS):
        int_ded = min(yearly_interest[y], SECTION24_LIMIT)
        prin_ded = min(yearly_principal[y], SECTION80C_LIMIT)
        total_tax_savings += (int_ded + prin_ded) * TAX_RATE

    net_emi_total = total_emi_paid - total_tax_savings
    avg_net_emi_month = net_emi_total / n_months

    # Property appreciation
    final_asset_value = future_value_lumpsum(price_num, APPRECIATION_RATE, LOAN_TENURE_YEARS)

    # Rental scenario for comparison
    initial_rent_monthly = price_num * RENT_YIELD / 12.0
    total_rent_20y = 0.0
    contributions = []  # monthly SIP contributions (difference between net EMI and rent, if positive)

    for year in range(LOAN_TENURE_YEARS):
        rent_monthly_this_year = initial_rent_monthly * ((1 + RENT_ESCALATION) ** year)
        total_rent_20y += rent_monthly_this_year * 12
        for _ in range(12):
            monthly_saving = max(avg_net_emi_month - rent_monthly_this_year, 0)
            contributions.append(monthly_saving)

    lumpsum_fv = future_value_lumpsum(down_payment, INVEST_RETURN_RATE, LOAN_TENURE_YEARS)
    sip_fv = future_value_sip(contributions, INVEST_RETURN_RATE, LOAN_TENURE_YEARS)

    final_wealth_buy = final_asset_value
    final_wealth_rent = lumpsum_fv + sip_fv

    recommendation = 'Buy'
    if final_wealth_rent > final_wealth_buy * 1.02:  # 2% threshold to avoid noise
        recommendation = 'Rent'
    elif abs(final_wealth_rent - final_wealth_buy) <= final_wealth_buy * 0.02:
        recommendation = 'Neutral'

    result.update({
        'Down Payment (INR)': down_payment,
        'Loan Amount (INR)': loan_amount,
        'Monthly EMI (INR)': emi,
        'Total Interest Paid (INR)': total_interest,
        'Total Principal Repaid (INR)': total_principal,
        'Total EMI Paid 20y (INR)': total_emi_paid,
        'Total Tax Savings 20y (INR)': total_tax_savings,
        'Net EMI Paid 20y (INR)': net_emi_total,
        'Avg Net EMI per Month (INR)': avg_net_emi_month,
        'Final Asset Value 20y (INR)': final_asset_value,
        'Initial Monthly Rent (INR)': initial_rent_monthly,
        'Total Rent Paid 20y (INR)': total_rent_20y,
        'Lumpsum FV from Down Payment (INR)': lumpsum_fv,
        'SIP FV from Monthly Savings (INR)': sip_fv,
        'Final Wealth - Buy (INR)': final_wealth_buy,
        'Final Wealth - Rent (INR)': final_wealth_rent,
        'Recommendation (Buy/Rent)': recommendation,
    })

    return result

def parse_card(card, listing_type):
    data = {}
    
    # Title
    title_tag = card.find('h2', class_='mb-srp__card--title')
    data['Property'] = clean_text(title_tag.get_text()) if title_tag else "N/A"
    
    # Price
    price_tag = card.find(class_='mb-srp__card__price')
    if price_tag:
        # The price tag often contains the rate as well (e.g. ₹ 1.6 Cr ₹ 4871 per sqft)
        # We try to separate them. Usually the first text node is the price.
        full_text = price_tag.get_text(strip=True)
        # Regex to find the main price (e.g. ₹ 1.6 Cr or ₹ 55 Lac)
        price_match = re.search(r'₹\s*[\d\.]+\s*(?:Cr|Lac|Lakh|Crore)', full_text, re.IGNORECASE)
        data['Price'] = price_match.group(0) if price_match else full_text
    else:
        data['Price'] = "Price on Request"

    # Bedrooms
    # Try to extract from title first (e.g. "3 BHK Flat")
    bhk_match = re.search(r'(\d+)\s*BHK', data['Property'], re.IGNORECASE)
    if bhk_match:
        data['Bedrooms'] = bhk_match.group(1)
    else:
        data['Bedrooms'] = "N/A"

    # Sale or Rent (set from the listing type passed in)
    data['For Sale/Rent'] = listing_type

    # Area
    # We look for summary items
    # Structure: div.mb-srp__card__summary__list -> div.mb-srp__card__summary__list--item
    # Inside item: div.mb-srp__card__summary--label and div.mb-srp__card__summary--value
    area_found = False
    
    # Strategy 1: Look for "Super Area" or "Carpet Area" labels
    summary_items = card.find_all(class_='mb-srp__card__summary--item') # This class might be different based on my analysis earlier
    # My analysis showed individual summary items text like "Super Area2610 sqft"
    # Let's try to find the label and value separately if they exist
    
    # Based on previous analysis:
    # Summary Item: Super Area2610 sqft...
    # It seems the label and value are siblings or parent/child.
    # Let's search for the label explicitly.
    
    labels = card.find_all(class_='mb-srp__card__summary--label')
    for label in labels:
        label_text = clean_text(label.get_text())
        if "Area" in label_text: # Super Area, Carpet Area
            # The value is usually the next sibling or in the same parent
            parent = label.parent
            value_tag = parent.find(class_='mb-srp__card__summary--value')
            if value_tag:
                data['Area of Property'] = clean_text(value_tag.get_text()) + " (" + label_text + ")"
                area_found = True
                break
    
    if not area_found:
        # Fallback: Search for "sqft" in text
        sqft_tag = card.find(text=re.compile("sqft", re.IGNORECASE))
        if sqft_tag:
            data['Area of Property'] = clean_text(sqft_tag)
        else:
            data['Area of Property'] = "N/A"

    return data

def scrape_listings(base_url, listing_type):
    all_listings = []
    
    for page in range(1, MAX_PAGES + 1):
        print(f"Scraping page {page}...")
        url = f"{base_url}&page={page}"
        
        try:
            response = requests.get(url, headers=HEADERS)
            if response.status_code != 200:
                print(f"Failed to retrieve page {page}. Status: {response.status_code}")
                continue
            
            soup = BeautifulSoup(response.text, 'html.parser')
            cards = soup.find_all('div', class_='mb-srp__card')
            
            if not cards:
                print(f"No more cards found on page {page}. Stopping.")
                break
                
            print(f"Found {len(cards)} listings on page {page}.")
            
            for card in cards:
                try:
                    listing_data = parse_card(card, listing_type)
                    all_listings.append(listing_data)
                except Exception as e:
                    print(f"Error parsing a card: {e}")
            
            # Be polite to the server
            time.sleep(1)
            
        except Exception as e:
            print(f"Error scraping page {page}: {e}")
            break
            
    return all_listings

def save_to_csv(listings):
    if not listings:
        print("No listings to save.")
        return

    # Convert to DataFrame for easier financial calculations
    df = pd.DataFrame(listings)

    # Ensure required base columns exist
    for col in ['Property', 'Area of Property', 'Price', 'Bedrooms', 'For Sale/Rent']:
        if col not in df.columns:
            df[col] = None

    # Compute financial metrics row-wise
    financial_rows = df.apply(compute_financials_row, axis=1, result_type='expand')
    df = pd.concat([df, financial_rows], axis=1)

    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print(f"Saved {len(df)} listings to {OUTPUT_FILE} with financial metrics.")

if __name__ == "__main__":
    print("Scraping SALE listings...")
    sale_listings = scrape_listings(BASE_URL_SALE, "Sale")

    print("Scraping RENT listings...")
    rent_listings = scrape_listings(BASE_URL_RENT, "Rent")

    all_listings = sale_listings + rent_listings
    save_to_csv(all_listings)
