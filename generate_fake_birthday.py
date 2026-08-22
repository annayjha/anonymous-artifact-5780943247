"""
Fake birthday generator
takes a birthday in Month D, YYYY format

python3 generate_fake_birthday.py --birthday "January 1, 1970"
"""

import argparse
from datetime import datetime
from faker import Faker

fake = Faker()
FMT = "%B %d, %Y"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--birthday", type=str)
    return parser.parse_args()


def get_fake_birthday(original_date):
    while True:
        new_date = fake.date_of_birth()
        if (
            new_date.month != original_date.month
            and new_date.day != original_date.day
            and new_date.year != original_date.year
            and new_date.year <= 2005
        ):
            return new_date


def main():
    args = parse_args()
    orig = datetime.strptime(args.birthday, FMT).date()
    new_date = get_fake_birthday(orig)
    print(new_date.strftime(FMT))


if __name__ == "__main__":
    main()
