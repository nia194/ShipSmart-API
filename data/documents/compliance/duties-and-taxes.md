# Import Duties and Taxes

When goods cross a border, the destination country may charge **import duties**
and **import taxes** (VAT or GST) before the shipment is released. These are
separate from the carrier's shipping charge and are set by the destination
country, not by ShipSmart or the carrier.

## What gets charged

- **Import duty** — a percentage of the goods' customs value, determined by the
  item's HS classification and the destination country's tariff schedule. See
  `hs-classification-basics.md` for how the HS code is chosen.
- **Import tax (VAT / GST)** — a consumption tax applied by many countries on
  the value of the goods plus, in most cases, the shipping and duty (a
  "tax-on-tax" base). Rates vary widely by country.
- **Brokerage / disbursement fees** — some carriers add a fee for advancing the
  duty/tax to customs on the recipient's behalf.

## De minimis

Most countries set a **de minimis** value below which duties and/or taxes are
waived. Shipments valued under that threshold clear without a duty/tax charge.
Thresholds differ per country and can change; see
`de-minimis-value-thresholds.md`. ShipSmart never advises splitting a shipment to
stay under a de minimis threshold — that is duty evasion (see the misuse policy).

## Landed cost

The **landed cost** is the total a buyer pays to get the goods delivered:
`goods value + shipping + duties + import taxes + fees`. A quote that shows only
the shipping price is not the landed cost. Whether duties/taxes are included in
what the sender pays depends on the Incoterm chosen — see `incoterms-ddp-ddu.md`.

## Who pays

By default, the **recipient** pays duties and taxes on delivery (DDU/DAP). The
sender can instead prepay them (DDP) so the recipient is not billed. This is a
commercial choice, not a compliance one — but the goods must still be declared
accurately regardless of who pays.

> Advisory only. ShipSmart explains how duties and taxes work and points to the
> destination country's rules; it does not compute a binding duty/tax amount or
> assert a shipment is "cleared". Exact figures come from the carrier/customs at
> the time of shipment.
