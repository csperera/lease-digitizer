"""
Generate 10 Realistic Government Leases for Demo
Based on GSA lease format with all fields populated
"""
import json
import random
from datetime import datetime, timedelta

# Realistic DC/Virginia government building addresses
PROPERTIES = [
    # Original 10
    {"address": "45 L Street, NE, Washington, DC 20002", "type": "Office Building", "name": "Sentinel Square III"},
    {"address": "1800 F Street, NW, Washington, DC 20405", "type": "Federal Office Building", "name": "GSA Headquarters"},
    {"address": "600 Army Navy Drive, Arlington, VA 22202", "type": "Secure Government Facility", "name": "Pentagon City Complex"},
    {"address": "1900 E Street, NW, Washington, DC 20415", "type": "Office Building", "name": "OPM Building"},
    {"address": "500 E Street, SW, Washington, DC 20024", "type": "Mixed-Use Federal Building", "name": "L'Enfant Plaza"},
    {"address": "2100 Crystal Drive, Arlington, VA 22202", "type": "Office Tower", "name": "Crystal City Tower"},
    {"address": "1201 Constitution Avenue, NW, Washington, DC 20230", "type": "Department Building", "name": "Commerce Department"},
    {"address": "400 7th Street, SW, Washington, DC 20590", "type": "Secure Facility", "name": "DOT Headquarters"},
    {"address": "1100 Ohio Drive, SW, Washington, DC 20242", "type": "Office Complex", "name": "Bureau of Engraving"},
    {"address": "955 L'Enfant Plaza, SW, Washington, DC 20024", "type": "Government Office Building", "name": "L'Enfant Tower"},
    # Additional 20
    {"address": "1800 M Street, NW, Washington, DC 20036", "type": "Class A Office Building", "name": "M Street Center"},
    {"address": "1255 23rd Street, NW, Washington, DC 20037", "type": "Office Tower", "name": "West End Tower"},
    {"address": "1775 Eye Street, NW, Washington, DC 20006", "type": "Premium Office Building", "name": "Eye Street Plaza"},
    {"address": "1620 L Street, NW, Washington, DC 20036", "type": "Mid-Rise Office", "name": "L Street Building"},
    {"address": "2000 Pennsylvania Avenue, NW, Washington, DC 20006", "type": "Mixed-Use Complex", "name": "Penn Square"},
    {"address": "4040 Wilson Boulevard, Arlington, VA 22203", "type": "Office Building", "name": "Ballston Tower"},
    {"address": "1550 Crystal Drive, Arlington, VA 22202", "type": "Office Complex", "name": "Crystal Park"},
    {"address": "2345 Crystal Drive, Arlington, VA 22202", "type": "Government Lease Building", "name": "Crystal Gateway"},
    {"address": "1101 Wilson Boulevard, Arlington, VA 22209", "type": "High-Rise Office", "name": "Rosslyn Plaza"},
    {"address": "1400 Key Boulevard, Arlington, VA 22209", "type": "Office Tower", "name": "Key Bridge Center"},
    {"address": "800 North Capitol Street, NW, Washington, DC 20002", "type": "Federal Building", "name": "Capitol Office"},
    {"address": "1310 G Street, NW, Washington, DC 20005", "type": "Office Building", "name": "Downtown Federal"},
    {"address": "1250 Connecticut Avenue, NW, Washington, DC 20036", "type": "Class A Office", "name": "Connecticut Plaza"},
    {"address": "1875 Connecticut Avenue, NW, Washington, DC 20009", "type": "Office Building", "name": "Dupont Center"},
    {"address": "3701 Pender Drive, Fairfax, VA 22030", "type": "Suburban Office Park", "name": "Fairfax Square"},
    {"address": "1650 Tysons Boulevard, McLean, VA 22102", "type": "Office Tower", "name": "Tysons Corner Center"},
    {"address": "8455 Colesville Road, Silver Spring, MD 20910", "type": "Office Complex", "name": "Silver Spring Metro"},
    {"address": "6116 Executive Boulevard, Rockville, MD 20852", "type": "Office Building", "name": "Executive Plaza"},
    {"address": "9800 Savage Road, Fort Meade, MD 20755", "type": "Secure Government Facility", "name": "Fort Meade Complex"},
    {"address": "1500 Pennsylvania Avenue, NW, Washington, DC 20220", "type": "Historic Federal Building", "name": "Treasury Annex"}
]

# Government tenants/agencies
TENANTS = [
    # Original 10
    "General Services Administration",
    "Department of Commerce",
    "Department of Transportation",
    "Office of Personnel Management",
    "Department of Homeland Security",
    "Department of Justice",
    "Department of State",
    "Department of Energy",
    "Environmental Protection Agency",
    "Social Security Administration",
    # Additional 20
    "Department of Defense",
    "Department of Health and Human Services",
    "Department of Veterans Affairs",
    "Department of the Interior",
    "Department of Agriculture",
    "Department of Labor",
    "Department of Education",
    "Department of the Treasury",
    "National Aeronautics and Space Administration",
    "Federal Bureau of Investigation",
    "Drug Enforcement Administration",
    "Bureau of Alcohol, Tobacco, Firearms and Explosives",
    "U.S. Marshals Service",
    "Secret Service",
    "Customs and Border Protection",
    "Federal Emergency Management Agency",
    "Transportation Security Administration",
    "U.S. Citizenship and Immigration Services",
    "National Science Foundation",
    "Small Business Administration"
]

# Landlord entities (typical commercial RE companies)
LANDLORDS = [
    # Original 10
    "Sentinel Square III, LLC",
    "Federal Realty Investment Trust",
    "Boston Properties, Inc.",
    "Carr Properties",
    "JBG SMITH Properties",
    "Brookfield Properties",
    "Tishman Speyer",
    "Hines Real Estate",
    "Columbia Property Trust",
    "Piedmont Office Realty Trust",
    # Additional 20
    "Vornado Realty Trust",
    "SL Green Realty Corp",
    "Kilroy Realty Corporation",
    "Maguire Properties",
    "Brandywine Realty Trust",
    "Douglas Development Corporation",
    "Monday Properties",
    "Clarion Partners",
    "CBRE Global Investors",
    "Beacon Capital Partners",
    "Shorenstein Properties",
    "Equity Office Properties",
    "Alexandria Real Estate Equities",
    "Paramount Group",
    "RXR Realty",
    "The Durst Organization",
    "Related Companies",
    "Silverstein Properties",
    "Forest City Realty Trust",
    "Akridge Development"
]


def generate_lease(index):
    """Generate a realistic government lease."""
    
    prop = PROPERTIES[index]
    
    # Realistic square footage (50K - 500K SF)
    rsf_options = [
        85000, 125000, 175000, 225000, 315000, 425000, 473000, 520000, 285000, 165000,
        95000, 145000, 195000, 265000, 335000, 385000, 445000, 490000, 305000, 155000,
        110000, 185000, 240000, 295000, 365000, 415000, 455000, 505000, 275000, 205000
    ]
    rsf = rsf_options[index]
    
    # ABOA typically 85-92% of RSF
    aboa_ratio = random.uniform(0.85, 0.92)
    aboa = int(rsf * aboa_ratio)
    
    # DC area government lease rates: $35-65/RSF
    rate_per_rsf = random.uniform(38.0, 62.0)
    annual_rent = rsf * rate_per_rsf
    
    # Parking: roughly 1 space per 1000 SF, government gets less
    parking_spaces = int(rsf / 2500) + random.randint(8, 24)
    parking_rate = random.choice([350, 375, 400, 425, 450])
    
    # Lease terms: government leases are typically 10-20 years
    lease_term_years = random.choice([10, 15, 20])
    
    # Generate dates
    commencement = datetime(2015, random.randint(1, 12), 1)
    expiration = commencement + timedelta(days=365 * lease_term_years)
    
    # Renewal options: typically 5-10 year options
    renewal_options = random.choice([
        "One 5-year renewal option",
        "Two 5-year renewal options", 
        "One 10-year renewal option",
        "Multiple 5-year renewal options available"
    ])
    
    # Operating expenses: $8-15/SF for DC area
    operating_expenses = random.uniform(9.0, 14.0)
    
    # Security deposit: typically 2-3 months rent
    monthly_rent = annual_rent / 12
    security_deposit = monthly_rent * random.choice([2, 3])
    
    # Lease number in GSA format
    lease_number = f"GS-11P-LDC{str(100 + index * 100).zfill(5)}"
    
    # Build the lease content (simulating extracted text)
    content = f"""LEASE NO. {lease_number}
Standard Lease GSA FORM L201C (May 2015)

This Lease is made and entered into between

{LANDLORDS[index]}

(Lessor), whose principal place of business is c/o {prop['name']}, Washington, DC,
and whose interest in the Property described herein is that of Fee Owner, and

The United States of America

(Government), acting by and through the designated representative of the General Services 
Administration (GSA), upon the terms and conditions set forth herein.

LEASE TERMS:

Property Address: {prop['address']}
Property Type: {prop['type']}

Space: {rsf:,} rentable square feet (RSF), yielding {aboa:,} ANSI/BOMA Office Area 
(ABOA) square feet (SF) of office and related space.

Tenant: {TENANTS[index]}

Annual Rent: ${annual_rent:,.2f} (${rate_per_rsf:.2f} per RSF)
Monthly Rent: ${monthly_rent:,.2f}

Operating Expenses: ${operating_expenses:.2f} per SF annually

Parking: {parking_spaces} reserved parking spaces at ${parking_rate:.2f} per space per month

Lease Term: {lease_term_years} years
Commencement Date: {commencement.strftime('%B %d, %Y')}
Expiration Date: {expiration.strftime('%B %d, %Y')}

Renewal Options: {renewal_options}

Security Deposit: ${security_deposit:,.2f}

Additional Terms:
- Base building services included
- Tenant improvement allowance provided
- Government shall have right to terminate for convenience
- Standard government lease provisions apply
- Property meets all federal security requirements
- LEED certification: Silver or higher

Special Provisions:
- Agency-specific security requirements applicable
- Government reserves right to install telecommunications equipment
- Parking allocation subject to change based on government needs
- Space may be used for government operations and related activities

The Government agrees to pay the Lessor the annual rent, payable in monthly installments 
in arrears, at the rate specified above. The annual rent shall be subject to operating 
expense and real estate tax adjustment during the lease term as outlined in this Lease.

This Lease may be renewed at the option of the Government for renewal terms as specified,
subject to agreement on rental rates and other terms at time of renewal.

Lessor shall furnish to the Government, as part of the rental consideration, all services,
improvements, alterations, repairs and utilities as defined by this Lease.
"""
    
    return {
        "filename": lease_number,
        "content": content
    }


def main():
    print("=" * 70)
    print("GENERATING 30 REALISTIC GOVERNMENT LEASES")
    print("=" * 70)
    print()
    
    leases = []
    
    for i in range(30):
        lease = generate_lease(i)
        leases.append(lease)
        
        # Show preview
        lines = lease['content'].split('\n')
        print(f"[{i+1}/30] {lease['filename']}")
        print(f"  Address: {PROPERTIES[i]['address']}")
        print(f"  Tenant: {TENANTS[i]}")
        print()
    
    # Save to JSON file
    output_file = "demo_leases_batch.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"leases": leases}, f, indent=2)
    
    print("=" * 70)
    print("LEASES GENERATED SUCCESSFULLY")
    print("=" * 70)
    print(f"Saved to: {output_file}")
    print(f"Total leases: {len(leases)}")
    print()
    print("NEXT STEP:")
    print("Run: python upload_demo_leases.py")
    print()


if __name__ == "__main__":
    main()
