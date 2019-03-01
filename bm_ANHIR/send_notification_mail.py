# -*- coding: utf-8 -*-
"""
Sending the invitation mail

"""

import os
import smtplib
from email.mime.text import MIMEText

import tqdm
import pandas as pd

# http://svti.fel.cvut.cz/cz/services/imap/client.html
SERVER = 'imap.feld.cvut.cz'
PORT = 25
LOGIN = 'username'  # univ user name
PASS = 'PASSWORD'  # my password
SENDER = 'Jiri Borovec <jiri.borovec@fel.cvut.cz>'
SUBJECT = 'ISBI 2019 - ANHIR - Image Registration Challenge'
# UPDATE_MAIL_TXT = 'mail_dataset.txt'
UPDATE_MAIL_TXT = 'mail_pre-release.txt'
MAIL_LIST_CSV = 'mail-list-test.csv'
# MAIL_LIST_CSV = 'mail-list.csv'


def load_text(name_file):
    with open(os.path.join(os.path.dirname(__file__), name_file)) as fp:
        text = fp.read()
    return text


def prepare_mail_invitation(name, pub, doi, link):
    """ prepare invitation mail with references

    :param str name: Author
    :param str pub: publication title
    :param str doi: publication DOI
    :param str link: publication link
    :return str:
    """
    text = load_text('mail_invitation.txt')
    text = text.replace('<NAME>', name)
    ref = 'https://www.doi.org/%s' % doi if isinstance(doi, str) else link
    text = text.replace('<PAPER-TITLE>', pub).replace('<PAPER-LINK>', ref)
    return text


def prepare_mail_update(name, mail_txt=UPDATE_MAIL_TXT):
    """ prepare general mail

    :param str name: Participant
    :return str:
    """
    text = load_text(mail_txt)
    text = text.replace('<NAME>', name)
    return text


def send_mail(smtp, email, row, subject=SUBJECT):
    """ send an email

    :param obj smtp:
    :param str email:
    :param {} row:
    :param str subject:
    """
    # text = prepare_mail_invitation(row['Name'], row['Publication name'],
    #                                row['DOI'], row['Publication link'])
    text = prepare_mail_update(row['Name'])
    # Open a plain text file for reading.  For this example, assume that
    # the text file contains only ASCII characters.
    mail = MIMEText(text)

    # me == the sender's email address
    # you == the recipient's email address
    mail['Subject'] = subject
    mail['From'] = SENDER
    # mail['To'] = '%s <%s>' % (name, email)
    mail['To'] = email

    # Send the message via our own SMTP server.
    smtp.sendmail(SENDER, email, mail.as_string())


def wrap_send_mail(idx, row, smtp):
    row = dict(row)
    try:
        send_mail(smtp, row['Email'], row)
    except Exception:
        print('ERROR: %i - %s - %s' % (idx, row['Name'], row['Email']))


def main(path_csv):
    """ main entry point

    :param str path_csv:
    """
    df_mail_list = pd.read_csv(path_csv)
    df_mail_list.drop_duplicates(subset=['Email'], inplace=True)
    print('Loaded items: %i' % len(df_mail_list))

    smtp = smtplib.SMTP(SERVER, PORT)
    smtp.starttls()
    smtp.login(LOGIN, PASS)
    print(repr(smtp))

    for idx, row in tqdm.tqdm(df_mail_list.iterrows(), desc='sending emails'):
        wrap_send_mail(idx, row, smtp)

    smtp.quit()
    print('Done :]')


if __name__ == '__main__':
    main(MAIL_LIST_CSV)
