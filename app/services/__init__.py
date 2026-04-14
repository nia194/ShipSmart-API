# TODO: Add service modules here as orchestration workflows are implemented.
# Services in this layer should:
#   - Coordinate calls to external APIs (LLMs, Java backend, Supabase)
#   - NOT own the system-of-record for transactional data
#   - Keep business logic thin until workflows are fully defined
#
# Example services to create:
#   - shipping_advisor.py  (AI shipping recommendations)
#   - address_validator.py (address validation workflow)
#   - tracking_advisor.py  (tracking issue analysis)
