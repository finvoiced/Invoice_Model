import spacy
import pandas as pd
from spacy import displacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_lg")
def NER(line_conf):
    for i in range(len(line_conf)):
        line = line_conf[i][0]
        doc = nlp(line)
        for ent in doc.ents:
            #print(ent.text, ent.start_char, ent.end_char, ent.label_)
            line_conf[i].append(ent.label_)
    return line_conf

# doc= nlp('Invoice, Invoice #: PREFIX-00044-2020-SUFFIX, Invoice Date: 2020-02-28 16:12:50, Order Date: 2020-01-20 10:14:19, Order Number: 5195, Payment Method: Credit card, Shipping Method: Shipping, ‘Couponts): Soff, Add some Terms & Conditions., Add some footer text., Hanzestraat 23,, TO06RH Doetinchem, The Netherlands, ‘CRN: $8789200, VAT ID: NLOO1115608871., Bill to:, Jane Doe Jane Doe, 25 Temple Road 25 Temple Road, (0X4 2€T Oxford (0X4 2€T Oxford, janedoe@mail.com, $50.00 1 $4.28 - $50.00, $40.00 1 - $7.88 $40.00, Discount (ex van) “$5.00, Shipping (ex van) $5.00, Fee (ex var) $3.00, Subtotal ex. var) $93.00, VAT 9% $4.28, vat 21% $9.56, Total (ine. vat) $106.84, Add some footer text.'
#          )
#
# print(displacy.render(doc, style="ent"))
#
# for e in doc.ents:
#     print(e.text, e.label_)