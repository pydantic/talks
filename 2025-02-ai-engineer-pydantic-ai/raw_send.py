import smtplib
from email.message import EmailMessage
from email.utils import make_msgid
import sys

if len(sys.argv) != 3:
    print('Usage: python raw_send.py <subject> <body>')
    sys.exit(1)

msg = EmailMessage()
msg['Subject'] = sys.argv[1]
msg['From'] = 'mail-server-test@helpmanual.io'
msg['To'] = 'spiced-ham@pydantic.io'
msg['Message-ID'] = make_msgid()
msg.set_content(sys.argv[2])

with smtplib.SMTP('route1.mx.cloudflare.net', 0) as server:
    server.starttls()  # Secure the connection
    server.send_message(msg)  # Send the email
    print('Email sent successfully.')
