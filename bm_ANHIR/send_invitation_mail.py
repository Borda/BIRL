# -*- coding: utf-8 -*-
"""
Sending the invitation mail

"""
import smtplib
from email.mime.text import MIMEText

import tqdm
import pandas as pd

# http://svti.fel.cvut.cz/cz/services/imap/client.html
SERVER = 'imap.feld.cvut.cz'
PORT = 25
LOGIN = 'borovji3'
PASS = 'XxXxX'
SENDER = 'Jiri Borovec <jiri.borovec@fel.cvut.cz>'
SUBJECT = 'ISBI 2019 - ANHIR - Image Registration Challenge'
MAIL_INVITATION = """
Dear <NAME>,

We are writing to you on behalf of your work on image registration and you work: “<PAPER-TITLE>” (<PAPER-LINK>).
We would like to invite you to participate in image registration methods to participate in the new Automatic Non-rigid Histological Image Registration (ANHIR) challenge to be held at the ISBI 2019 conference in Venice in April 2019.
The task consists of registering multi-stain histology images. For more detail visit our webpage:

https://anhir.grand-challenge.org

Looking forward to your possible participation,
Jiří Borovec & Arrate Munoz-Barrutia & Jan Kybic & Ignacio Arganda
"""


def send_mail_invitation(name, email, pub, doi, link, smtp):
    text = MAIL_INVITATION.replace('<NAME>', name)
    ref = 'https://www.doi.org/%s' % doi if isinstance(doi, str) else link
    text = text.replace('<PAPER-TITLE>', pub).replace('<PAPER-LINK>', ref)

    # Open a plain text file for reading.  For this example, assume that
    # the text file contains only ASCII characters.
    mail = MIMEText(text)

    # me == the sender's email address
    # you == the recipient's email address
    mail['Subject'] = SUBJECT
    mail['From'] = SENDER
    # mail['To'] = '%s <%s>' % (name, email)
    mail['To'] = email

    # Send the message via our own SMTP server.
    smtp.sendmail(SENDER, email, mail.as_string())


def send_mail(idx, row, smtp):
    row = dict(row)
    try:
        send_mail_invitation(row['Name'], row['Email'],
                             row['Publication name'],
                             row['DOI'], row['Publication link'],
                             smtp)
    except Exception:
        print('ERROR: %i - %s - %s' % (idx, row['Name'], row['Email']))


def main(path_csv):
    df_invit = pd.read_csv(path_csv)
    print('Loaded items: %i' % len(df_invit))

    smtp = smtplib.SMTP(SERVER, PORT)
    smtp.starttls()
    smtp.login(LOGIN, PASS)
    print(repr(smtp))

    for idx, row in tqdm.tqdm(df_invit.iterrows()):
        send_mail(idx, row, smtp)

    smtp.quit()
    print('Done :]')


if __name__ == '__main__':
    main('mail-list-test.csv')
