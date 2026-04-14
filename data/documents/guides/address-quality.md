# Address Quality and Delivery Issues

## Why Address Quality Matters
Address errors are the #1 cause of failed deliveries and package returns. An estimated 5-10% of shipping addresses contain errors that can cause delays, misdelivery, or returns. Validating addresses before shipping saves time, money, and customer frustration.

## Common Address Problems

### Missing or Incomplete Information
- Missing apartment, suite, or unit number (most common issue)
- Missing directional prefix/suffix (N, S, E, W)
- Abbreviated street names that are ambiguous
- Missing city or state

### Formatting Issues
- Misspelled street names or city names
- Wrong ZIP code for the city/state combination
- Using old/outdated street names (after city renaming)
- Using informal neighborhood names instead of official city names

### Invalid Addresses
- Address does not exist (wrong house number)
- Address is a vacant lot or demolished building
- PO Box number that doesn't exist
- Address in a gated community without gate code

## Residential vs Commercial Addresses

### Why It Matters
- UPS and FedEx charge a residential surcharge ($4-6 per package)
- USPS does not charge a residential surcharge
- Some carriers have different service levels for residential (e.g., FedEx Home Delivery)
- Delivery windows differ: commercial addresses typically receive morning delivery; residential may be afternoon/evening

### How to Tell
- Most address validation APIs (UPS, FedEx, USPS) classify addresses as residential or commercial
- Multi-tenant buildings may be classified differently depending on the suite/unit
- Home-based businesses are typically classified as residential by carriers

### Impact on Shipping Cost
For high-volume shippers, residential surcharges add up significantly. Strategies:
- Use USPS for residential deliveries (no surcharge)
- Offer pickup at commercial locations or lockers
- Negotiate residential surcharge discounts with carrier accounts

## Failed Delivery Prevention

### Before Shipping
1. Validate the address using carrier address validation APIs
2. Verify apartment/suite/unit numbers are included
3. Confirm the ZIP code matches the city and state
4. For international: verify the postal code format for the destination country

### Common Causes of Failed Delivery
- Nobody home to receive the package (if signature required)
- Incorrect apartment/unit number
- Address not found by driver
- Access issues (gated community, locked building, aggressive dog)
- Weather or natural disaster

### What Happens After Failed Delivery
1. **First attempt fails**: Carrier leaves a notice and schedules redelivery
2. **Second attempt fails**: Another notice; package may be held at local facility
3. **Third attempt fails**: Package is returned to sender (UPS/FedEx) or held for pickup (USPS)
4. Return shipping is charged to the sender

### Delivery Instructions
Most carriers support delivery instructions:
- "Leave at front door" / "Leave at back door"
- "Leave with neighbor"
- "Hold at facility for pickup"
- Gate codes or building access instructions
- These can be set through carrier delivery management tools (UPS My Choice, FedEx Delivery Manager)

## International Address Considerations
- Address formats vary significantly by country
- Some countries use postal codes, others don't
- Street naming conventions differ (house number before or after street name)
- Always include country code and full postal code
- Use the destination country's address format, not the US format
