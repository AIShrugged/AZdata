-- AZdata e-Invoice · Task 1 schema (PostgreSQL 16)
-- This DDL is the SOURCE the metadata-catalog parser consumes:
--   * table/column structure  -> parsed from CREATE TABLE
--   * business concept (EN)    -> carried as COMMENT ON COLUMN
--   * AZ/EN synonyms + roles   -> added from config/metadata_enrichment.yaml
-- Keep column COMMENTs in sync with the brief's metadata-catalog examples.

CREATE TABLE IF NOT EXISTS taxpayer (
    tin   text PRIMARY KEY,
    name  text
);
COMMENT ON TABLE  taxpayer      IS 'Taxpayer directory (TIN -> name)';
COMMENT ON COLUMN taxpayer.tin  IS 'Taxpayer identification number (VOEN)';
COMMENT ON COLUMN taxpayer.name IS 'Taxpayer name';

CREATE TABLE IF NOT EXISTS einvoice (
    id                       bigserial PRIMARY KEY,
    supplier_tin             text   NOT NULL,
    recipient_tin            text   NOT NULL,
    einvoice_date            date   NOT NULL,
    approval_date            date,
    series                   text,
    number                   bigint,
    excise_amount            numeric(18,2) NOT NULL DEFAULT 0,
    vat_taxable_amount       numeric(18,2) NOT NULL DEFAULT 0,
    non_vat_taxable_amount   numeric(18,2) NOT NULL DEFAULT 0,
    vat_exempt_amount        numeric(18,2) NOT NULL DEFAULT 0,
    zero_rated_amount        numeric(18,2) NOT NULL DEFAULT 0,
    vat_amount               numeric(18,2) NOT NULL DEFAULT 0,
    road_tax                 numeric(18,2) NOT NULL DEFAULT 0,
    total_amount             numeric(18,2) NOT NULL DEFAULT 0,
    CONSTRAINT fk_einvoice_supplier  FOREIGN KEY (supplier_tin)  REFERENCES taxpayer(tin),
    CONSTRAINT fk_einvoice_recipient FOREIGN KEY (recipient_tin) REFERENCES taxpayer(tin)
);
COMMENT ON TABLE  einvoice                        IS 'Electronic invoice header';
COMMENT ON COLUMN einvoice.supplier_tin           IS 'Submitting taxpayer — TIN of the e-invoice issuer';
COMMENT ON COLUMN einvoice.recipient_tin          IS 'Receiving taxpayer — TIN of the e-invoice recipient';
COMMENT ON COLUMN einvoice.einvoice_date          IS 'Invoice date — date of the e-invoice';
COMMENT ON COLUMN einvoice.approval_date          IS 'Approval date of the e-invoice';
COMMENT ON COLUMN einvoice.series                 IS 'E-invoice series';
COMMENT ON COLUMN einvoice.number                 IS 'E-invoice number';
COMMENT ON COLUMN einvoice.excise_amount          IS 'Excise amount';
COMMENT ON COLUMN einvoice.vat_taxable_amount     IS 'Amount of VAT-taxable transactions';
COMMENT ON COLUMN einvoice.non_vat_taxable_amount IS 'Amount of non-VAT-taxable transactions';
COMMENT ON COLUMN einvoice.vat_exempt_amount      IS 'Amount of VAT-exempt transactions';
COMMENT ON COLUMN einvoice.zero_rated_amount      IS 'Amount of zero-rated VAT transactions';
COMMENT ON COLUMN einvoice.vat_amount             IS 'VAT amount';
COMMENT ON COLUMN einvoice.road_tax               IS 'Road tax';
COMMENT ON COLUMN einvoice.total_amount           IS 'Turnover — total amount';

CREATE INDEX IF NOT EXISTS ix_einvoice_supplier  ON einvoice(supplier_tin);
CREATE INDEX IF NOT EXISTS ix_einvoice_recipient ON einvoice(recipient_tin);
CREATE INDEX IF NOT EXISTS ix_einvoice_date      ON einvoice(einvoice_date);
