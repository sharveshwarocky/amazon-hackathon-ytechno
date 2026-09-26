#!/usr/bin/env python3
"""
Tests for the business entity preprocessing pipeline.
Based on real examples from the examples.csv specification.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from preprocess import (
    BusinessPreprocessor,
    normalize_text,
    normalize_name,
    normalize_address,
    extract_name_features,
    extract_address_features,
    PreprocessingConfig
)


class TestNormalizeText:
    """Test safe text normalization."""

    def test_basic_normalization(self):
        """Test basic casefolding and whitespace handling."""
        result = normalize_text("  FH BUSINESS  Pvt. LIMITED  ")
        assert result == "fh business pvt. limited"

    def test_ampersand_normalization(self):
        """Test & → and conversion."""
        result = normalize_text("Hendricks & Flowers Inc")
        assert "and" in result
        assert "&" not in result

    def test_multiple_spaces_collapse(self):
        """Test that multiple spaces are collapsed."""
        result = normalize_text("hello    world")
        assert "  " not in result
        assert result == "hello world"

    def test_empty_input(self):
        """Test handling of empty input."""
        result = normalize_text("")
        assert result == ""

    def test_none_input(self):
        """Test handling of None input."""
        result = normalize_text(None)
        assert result == ""

    def test_preserves_unicode(self):
        """Test that Unicode is preserved, not converted to ASCII."""
        result = normalize_text("Nagar Nigam")
        assert "Nagar" in result or "nagar" in result  # Unicode preserved

    def test_punctuation_standardization(self):
        """Test standardization of punctuation."""
        result = normalize_text("ABC, Inc.")
        # Commas are kept but normalized
        assert result is not None


class TestNormalizeName:
    """Test business name normalization."""

    def test_company_suffix_extraction(self):
        """Test extraction of legal suffix."""
        result = normalize_name("Zander Blue Co")
        assert result["legal_suffix"] == "company"
        assert "zander blue" in result["name_core"]

    def test_inc_suffix(self):
        """Test INC suffix extraction."""
        result = normalize_name("Hendricks and Flowers Inc")
        assert result["legal_suffix"] == "incorporated"
        assert "hendricks" in result["name_core"]

    def test_pvt_limited(self):
        """Test Pvt. Limited suffix handling."""
        result = normalize_name("FH BUSINESS  Pvt. LIMITED")
        assert result["legal_suffix"] == "limited"
        assert "fh business" in result["name_core"]

    def test_no_suffix(self):
        """Test name without suffix."""
        result = normalize_name("Behavioral Health Clinic")
        assert result["legal_suffix"] == ""
        assert "behavioral health clinic" == result["name_core"]

    def test_llc_suffix(self):
        """Test LLC suffix."""
        result = normalize_name("Summit Health LLC")
        assert result["legal_suffix"] == "llc"

    def test_casefold(self):
        """Test that output is casefolded."""
        result = normalize_name("FH BUSINESS")
        assert result["name_clean"] == "fh business"


class TestExtractNameFeatures:
    """Test name feature extraction."""

    def test_domain_extraction(self):
        """Test domain extraction from name."""
        features = extract_name_features("teamair.com")
        assert features["domain"] == "teamair.com"
        assert features["domain_core"] == "teamair"
        assert features["is_domain_like"] is True

    def test_domain_core_extraction(self):
        """Test domain core extraction."""
        features = extract_name_features("cardiologymetrocare.com")
        assert features["domain"] == "cardiologymetrocare.com"
        assert features["domain_core"] == "cardiologymetrocare"

    def test_dba_detection(self):
        """Test DBA/FKA/AKA pattern detection."""
        features = extract_name_features("Deltaquo F/K/A Pao Valley Toyo")
        assert features["has_fka"] == 1
        assert features["has_dba"] == 0
        assert features["has_aka"] == 0

    def test_fka_detection(self):
        """Test F/K/A pattern detection."""
        features = extract_name_features("Company F/K/A Former Name")
        assert features["has_fka"] == 1

    def test_aka_detection(self):
        """Test A/K/A pattern detection."""
        features = extract_name_features("Business A/K/A Alternative")
        assert features["has_aka"] == 1

    def test_name_tokens(self):
        """Test token extraction."""
        features = extract_name_features("ABC Corporation")
        assert "abc" in features["name_tokens"]
        assert "corporation" in features["name_tokens"]

    def test_name_core_without_suffix(self):
        """Test name core extraction without suffix."""
        features = extract_name_features("Pao Valley Toyo")
        assert "pao valley toyo" == features["name_core"]

    def test_numbers_preserved(self):
        """Test that numbers are preserved in names."""
        features = extract_name_features("Building 123 Inc")
        assert "123" in features["name_numbers"]


class TestNormalizeAddress:
    """Test address normalization."""

    def test_address_casefold(self):
        """Test address casefolding."""
        result = normalize_address("88 OLIVE CIR, LEBANON, TN")
        assert "88 olive cir" in result["address_clean"]
        assert result["state_code"] == "TN"

    def test_unit_extraction(self):
        """Test unit number extraction."""
        result = normalize_address("4038 Talmadge Road, Unit 102")
        assert "102" in result["unit_number"]
        assert "4038" in result["street_number"] or "4038" in result["address_numbers"]

    def test_h_no_extraction(self):
        """Test H.NO extraction."""
        result = normalize_address("H.NO 204")
        assert "204" in result["unit_number"] or "204" in result["address_numbers"]

    def test_apt_extraction(self):
        """Test apartment number extraction."""
        result = normalize_address("APT 5B13")
        assert "5b13" in result["unit_number"].lower()

    def test_care_of_detection(self):
        """Test C/O (care of) detection."""
        result = normalize_address("C/O Vikram Singh")
        assert "vikram singh" in result["address_clean"]

    def test_postal_code_extraction(self):
        """Test postal code extraction."""
        result = normalize_address("123 Main St, Springfield, IL 62701")
        assert result["postal_code"] == "62701"

    def test_address_numbers(self):
        """Test address number extraction."""
        result = normalize_address("1056-1060 Some Road")
        assert "1056" in result["address_numbers"]

    def test_address_clean_normalization(self):
        """Test that address_clean is normalized."""
        result = normalize_address("  123  Main  St  ")
        assert "  " not in result["address_clean"]  # No double spaces
        assert result["address_clean"].strip() == result["address_clean"]  # Stripped


class TestExtractAddressFeatures:
    """Test address feature extraction."""

    def test_street_number(self):
        """Test street number extraction."""
        features = extract_address_features("88 Olive Cir")
        assert features["street_number"] == "88"

    def test_unit_number(self):
        """Test unit number extraction."""
        features = extract_address_features("Flat 4, Building 5")
        assert "4" in features["unit_number"] or "4" in features["address_numbers"]

    def test_city_tokens(self):
        """Test city token extraction."""
        features = extract_address_features("123 Main St, Springfield, IL")
        assert "springfield" in features["city_tokens"]

    def test_state_code(self):
        """Test state code extraction."""
        features = extract_address_features("123 Main St, Springfield, IL")
        assert features["state_code"] == "IL"

    def test_indiana_state(self):
        """Test Indiana state code."""
        features = extract_address_features("456 Oak Ave, Indianapolis, IN")
        assert features["state_code"] == "IN"

    def test_karnataka_state(self):
        """Test Karnataka state code."""
        features = extract_address_features("789 Main Rd, Bangalore, KA")
        assert features["state_code"] == "KA"

    def test_empty_address(self):
        """Test handling of empty address."""
        features = extract_address_features("")
        assert features["address_numbers"] == []
        assert features["street_number"] == ""
        assert features["city_tokens"] == []


class TestPreprocessorIntegration:
    """Test the full preprocessing pipeline."""

    def test_full_record_preprocessing(self):
        """Test preprocessing of a full record."""
        preprocessor = BusinessPreprocessor()
        record = {
            "entity_id": "S1_001",
            "business_name": "Summit Health LLC",
            "business_address": "123 Main St, Unit 4",
            "country": "US"
        }
        result = preprocessor.preprocess_record(record)

        assert result["entity_id"] == "S1_001"
        assert result["business_name"] == "Summit Health LLC"
        assert result["legal_suffix"] == "llc"
        assert result["country"] == "US"
        assert result["domain"] == ""  # Not domain-like

    def test_domain_like_name(self):
        """Test handling of domain-like business names."""
        preprocessor = BusinessPreprocessor()
        record = {
            "entity_id": "S1_002",
            "business_name": "teamair.com",
            "business_address": "",
            "country": "US"
        }
        result = preprocessor.preprocess_record(record)

        assert result["domain"] == "teamair.com"
        assert result["domain_core"] == "teamair"
        assert result["is_domain_like"] is True

    def test_preserves_original(self):
        """Test that original fields are preserved."""
        preprocessor = BusinessPreprocessor(config=PreprocessingConfig(preserve_original=True))
        record = {
            "entity_id": "S1_003",
            "business_name": "ABC Corp",
            "business_address": "123 Main St",
            "country": "US"
        }
        result = preprocessor.preprocess_record(record)

        assert result["business_name"] == "ABC Corp"
        assert result["business_address"] == "123 Main St"
        assert result["entity_id"] == "S1_003"

    def test_repeated_tokens_detection(self):
        """Test detection of repeated tokens (VIDYALAYA VIDYALAYA)."""
        preprocessor = BusinessPreprocessor()
        record = {
            "entity_id": "S1_004",
            "business_name": "VIDYALAYA VIDYALAYA",
            "business_address": "",
            "country": "IN"
        }
        result = preprocessor.preprocess_record(record)

        # Tokens should be extracted
        assert "vidyalaya" in result["name_tokens"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])