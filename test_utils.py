# test_modelo.py
from src.utils import (
    get_colum_by_type,
    has_rows,
    read_file,
    read_configuration,
    check_columns
)

def test_config_file():
    """Test if config file exist and can be parse"""
    assert isinstance(read_configuration(),dict)
