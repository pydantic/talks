"""Evaluation dataset and evaluators for contact extraction.

This module defines:
- ContactCaseMetadata: Metadata for each test case
- FieldAccuracyEvaluator: Evaluates per-field extraction accuracy
- contact_dataset: The evaluation dataset
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from task import ContactInfo, TaskInput


@dataclass
class ContactCaseMetadata:
    """Metadata for contact extraction test cases."""

    difficulty: str  # 'easy', 'medium', 'hard'
    has_noise: bool  # Whether the text contains irrelevant information
    description: str  # What this case tests


@dataclass
class FieldAccuracyEvaluator(Evaluator[TaskInput, ContactInfo, ContactCaseMetadata]):
    """Evaluates how many fields were correctly extracted.

    Returns a score between 0.0 and 1.0 based on field-level accuracy.
    """

    def evaluate(self, ctx: EvaluatorContext[TaskInput, ContactInfo, ContactCaseMetadata]) -> dict[str, Any]:
        if ctx.expected_output is None:
            return {'accuracy': 1.0}

        expected = ctx.expected_output
        output = ctx.output

        # Count correct fields
        fields = ['name', 'email', 'phone', 'company', 'title']
        correct = 0
        total = 0

        field_results = {}
        for field in fields:
            expected_val = getattr(expected, field)
            output_val = getattr(output, field)

            # Only count fields that have expected values
            if expected_val is not None:
                total += 1
                # Normalize comparison (case-insensitive, strip whitespace)
                expected_norm = str(expected_val).lower().strip()
                output_norm = str(output_val).lower().strip() if output_val else ''

                is_correct = (
                    expected_norm == output_norm or expected_norm in output_norm or output_norm in expected_norm
                )
                if is_correct:
                    correct += 1
                field_results[f'{field}_correct'] = is_correct

        accuracy = correct / total if total > 0 else 1.0

        return {
            'accuracy': accuracy,
            'fields_correct': correct,
            'fields_total': total,
            **field_results,
        }


# Define test cases for contact extraction
contact_cases: list[Case[TaskInput, ContactInfo, ContactCaseMetadata]] = [
    # Easy cases - straightforward contact information
    Case(
        name='simple_email_signature',
        inputs=TaskInput(text='Best regards,\nJohn Smith\njohn.smith@example.com\n555-123-4567'),
        expected_output=ContactInfo(
            name='John Smith',
            email='john.smith@example.com',
            phone='555-123-4567',
        ),
        metadata=ContactCaseMetadata(
            difficulty='easy',
            has_noise=False,
            description='Simple email signature format',
        ),
    ),
    Case(
        name='business_card_format',
        inputs=TaskInput(
            text='Jane Doe\nSenior Software Engineer\nAcme Corporation\njane.doe@acme.com\n(415) 555-0100'
        ),
        expected_output=ContactInfo(
            name='Jane Doe',
            email='jane.doe@acme.com',
            phone='(415) 555-0100',
            company='Acme Corporation',
            title='Senior Software Engineer',
        ),
        metadata=ContactCaseMetadata(
            difficulty='easy',
            has_noise=False,
            description='Standard business card format',
        ),
    ),
    # Medium cases - some ambiguity or formatting variations
    Case(
        name='inline_contact',
        inputs=TaskInput(
            text='For more information, contact Michael Johnson at michael.j@techcorp.io or call him at 1-800-TECH-123.'
        ),
        expected_output=ContactInfo(
            name='Michael Johnson',
            email='michael.j@techcorp.io',
            phone='1-800-TECH-123',
        ),
        metadata=ContactCaseMetadata(
            difficulty='medium',
            has_noise=True,
            description='Contact info embedded in a sentence',
        ),
    ),
    Case(
        name='international_format',
        inputs=TaskInput(
            text='Dr. Maria Garcia\nHead of Research\nGlobal Health Institute\nmgarcia@ghi.org\n+44 20 7946 0958'
        ),
        expected_output=ContactInfo(
            name='Maria Garcia',
            email='mgarcia@ghi.org',
            phone='+44 20 7946 0958',
            company='Global Health Institute',
            title='Head of Research',
        ),
        metadata=ContactCaseMetadata(
            difficulty='medium',
            has_noise=False,
            description='International phone format and title prefix',
        ),
    ),
    # Hard cases - noisy text, unusual formats
    Case(
        name='noisy_email_thread',
        inputs=TaskInput(
            text="""Re: Meeting Follow-up

Hi team,

Thanks for joining today's call. As discussed, please reach out to our new vendor contact:

Robert Chen
VP of Sales, CloudTech Solutions
r.chen@cloudtech.solutions
Mobile: +1 (650) 555-8900

He'll be handling our account going forward. The previous contact (sarah@oldvendor.com) is no longer with the company.

Best,
Alex"""
        ),
        expected_output=ContactInfo(
            name='Robert Chen',
            email='r.chen@cloudtech.solutions',
            phone='+1 (650) 555-8900',
            company='CloudTech Solutions',
            title='VP of Sales',
        ),
        metadata=ContactCaseMetadata(
            difficulty='hard',
            has_noise=True,
            description='Email thread with multiple contacts, need to identify the main one',
        ),
    ),
    Case(
        name='partial_info',
        inputs=TaskInput(text='Please contact support at help@service.io for assistance.'),
        expected_output=ContactInfo(
            email='help@service.io',
        ),
        metadata=ContactCaseMetadata(
            difficulty='medium',
            has_noise=False,
            description='Only email available, no name or phone',
        ),
    ),
    Case(
        name='informal_intro',
        inputs=TaskInput(text="Hey! I'm Sam from StartupXYZ. Hit me up at sam@startupxyz.com or text 555-STARTUP"),
        expected_output=ContactInfo(
            name='Sam',
            email='sam@startupxyz.com',
            phone='555-STARTUP',
            company='StartupXYZ',
        ),
        metadata=ContactCaseMetadata(
            difficulty='medium',
            has_noise=False,
            description='Informal language and vanity phone number',
        ),
    ),
    Case(
        name='complex_signature',
        inputs=TaskInput(
            text="""--
Emily Watson, Ph.D.
Director of Engineering | AI Division
TechGiant Inc. (NASDAQ: TGNT)

Email: e.watson@techgiant.com
Office: (555) 100-2000 ext. 4501
LinkedIn: linkedin.com/in/emilywatson

"Innovation distinguishes between a leader and a follower."
This email may contain confidential information..."""
        ),
        expected_output=ContactInfo(
            name='Emily Watson',
            email='e.watson@techgiant.com',
            phone='(555) 100-2000 ext. 4501',
            company='TechGiant Inc.',
            title='Director of Engineering',
        ),
        metadata=ContactCaseMetadata(
            difficulty='hard',
            has_noise=True,
            description='Complex signature with credentials, stock ticker, and noise',
        ),
    ),
]

# Create the evaluation dataset
contact_dataset: Dataset[TaskInput, ContactInfo, ContactCaseMetadata] = Dataset(
    name='contact_extraction',
    cases=contact_cases,
    evaluators=[FieldAccuracyEvaluator()],
)
