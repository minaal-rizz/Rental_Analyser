import os
import random
import csv
import json
from faker import Faker
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from datetime import datetime, timedelta

# Initialize Faker instance for generating synthetic names and dates
fake = Faker()

# Directory where rental agreement PDFs will be saved
OUTPUT_DIR = "data/rentals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List to store metadata for all generated rental agreements
metadata_list = []

# Predefined list of cities for rental agreements
cities = ["Lahore", "Karachi", "Islamabad", "Faisalabad", "Multan", "Peshawar", "Quetta"]


def generate_rental_agreement(branch_id):
    """
    Generate a synthetic rental agreement PDF and return its metadata.

    Args:
        branch_id (int): Unique identifier for the branch.

    Returns:
        dict: Metadata about the generated rental agreement including:
            - branch_id (int): Unique branch identifier.
            - branch_name (str): Branch name (Branch-{id} format).
            - city (str): Randomly chosen city.
            - landlord (str): Fake landlord name.
            - yearly_rent (int): Rent amount in PKR.
            - agreement_start (str): Start date of the agreement.
            - expiry_date (str): Expiry date of the agreement.
            - duration_years (int): Duration of the agreement in years.
            - file_path (str): Path to the generated PDF.

    Notes:
        - PDF files are created using `reportlab`.
        - Metadata is structured for later use in retrieval/analysis pipelines.
    """
    branch_name = f"Branch-{branch_id}"
    city = random.choice(cities)
    landlord = fake.name()
    yearly_rent = random.randint(500000, 5000000)  # yearly rent in PKR
    agreement_start = fake.date_between(start_date="-5y", end_date="today")
    duration_years = random.randint(1, 5)
    expiry_date = agreement_start + timedelta(days=365 * duration_years)

    # File path for the generated PDF
    file_name = f"rental_agreement_{branch_id}.pdf"
    file_path = os.path.join(OUTPUT_DIR, file_name)

    # Generate PDF using ReportLab
    c = canvas.Canvas(file_path, pagesize=LETTER)
    c.setFont("Helvetica", 12)

    # Agreement text lines
    text_lines = [
        f"Rental Agreement - {branch_name}",
        f"City: {city}",
        f"Landlord: {landlord}",
        f"Yearly Rent: PKR {yearly_rent:,}",
        f"Agreement Start: {agreement_start}",
        f"Expiry Date: {expiry_date}",
        f"Duration: {duration_years} years",
        "Terms & Conditions: This is a synthetic rental agreement for testing purposes only."
    ]

    # Write text to PDF
    y = 750
    for line in text_lines:
        c.drawString(50, y, line)
        y -= 20

    c.save()

    # Store metadata for the generated agreement
    metadata = {
        "branch_id": branch_id,
        "branch_name": branch_name,
        "city": city,
        "landlord": landlord,
        "yearly_rent": yearly_rent,
        "agreement_start": str(agreement_start),
        "expiry_date": str(expiry_date),
        "duration_years": duration_years,
        "file_path": file_path
    }

    return metadata


def main(total_files=1100):
    """
    Generate multiple rental agreements and save their metadata.

    Args:
        total_files (int, optional): Number of agreements to generate. Defaults to 1100.

    Actions:
        - Generates rental agreement PDFs.
        - Collects metadata for each agreement.
        - Saves metadata to both CSV and JSON formats.

    Example:
        >>> main(100)
        ✅ Generated 100 agreements in data/rentals
        ✅ Metadata saved to data/rental_metadata.csv and data/rental_metadata.json
    """
    print(f"Generating {total_files} rental agreements...")
    for i in range(1, total_files + 1):
        metadata = generate_rental_agreement(i)
        metadata_list.append(metadata)

    # Save metadata to CSV
    csv_file = os.path.join("data", "rental_metadata.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metadata_list[0].keys())
        writer.writeheader()
        writer.writerows(metadata_list)

    # Save metadata to JSON
    json_file = os.path.join("data", "rental_metadata.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=4)

    print(f"✅ Generated {total_files} agreements in {OUTPUT_DIR}")
    print(f"✅ Metadata saved to {csv_file} and {json_file}")


if __name__ == "__main__":
    main(1100)  # generate 1100 agreements
