from time import sleep

import stripe
import logfire
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from opentelemetry.instrumentation.requests import RequestsInstrumentor

logfire.configure()

# you can get a testing key by just signing up for stripe
STRIPE_KEY = 'sk_test_...'
stripe.api_key = STRIPE_KEY
RequestsInstrumentor().instrument()

app = FastAPI()
logfire.instrument_fastapi(app)
logfire.info('Starting the app')


RequestsInstrumentor().instrument()


@app.post('/payments/{user_id:int}/complete/')
def hello(user_id: int):
    amount, currency, payment_method = get_payment_details(user_id)

    try:
        intent = stripe.PaymentIntent.create(
            amount=amount,
            currency=currency,
            payment_method=payment_method,
            confirm=True,
            return_url='https://example.com/return',
        )
    except stripe.CardError:
        store_payment_failure(user_id)
        return JSONResponse(content={'detail': 'Card error'}, status_code=400)
    else:
        store_payment_success(user_id, intent)


@logfire.instrument()
def get_payment_details(user_id: int) -> [int, str, str]:
    sleep(0.2)
    if user_id == 42:
        return 20_00, 'usd', 'pm_card_visa'
    else:
        return 20_00, 'usd', 'pm_card_visa_chargeDeclinedInsufficientFunds'


@logfire.instrument()
def store_payment_success(user_id: int, _indent) -> None:
    sleep(0.2)


@logfire.instrument()
def store_payment_failure(user_id: int) -> None:
    sleep(0.2)
