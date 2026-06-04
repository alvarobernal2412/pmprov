from tracker.delta_calculator import compute_delta


def df_snap(oid, shape, columns, dtypes):
    return {"kind": "dataframe", "object_id": oid,
            "shape": shape, "columns": columns, "dtypes": dtypes}

def arr_snap(oid, shape, dtype):
    return {"kind": "array", "object_id": oid,
            "shape": shape, "dtype": dtype, "nbytes": 100}

def col_snap(oid, length):
    return {"kind": "collection", "object_id": oid,
            "collection_type": "list", "length": length}


# --- empty / unknown ---

def test_empty_pre_returns_unknown():
    assert compute_delta({}, {"kind": "dataframe"}) == {"kind": "unknown"}

def test_empty_post_returns_unknown():
    assert compute_delta({"kind": "dataframe"}, {}) == {"kind": "unknown"}


# --- mutation_type ---

def test_same_object_id_is_in_place():
    pre = df_snap(1, (10, 3), ["a", "b", "c"], {"a": "int64", "b": "float64", "c": "object"})
    post = df_snap(1, (10, 4), ["a", "b", "c", "d"], {"a": "int64", "b": "float64", "c": "object", "d": "bool"})
    assert compute_delta(pre, post)["mutation_type"] == "in_place"

def test_different_object_id_is_derivation():
    pre = df_snap(1, (10, 2), ["a", "b"], {"a": "int64", "b": "float64"})
    post = df_snap(2, (8, 2), ["a", "b"], {"a": "int64", "b": "float64"})
    assert compute_delta(pre, post)["mutation_type"] == "derivation"


# --- dataframe delta fields ---

def test_columns_added():
    pre = df_snap(1, (10, 2), ["a", "b"], {"a": "int64", "b": "float64"})
    post = df_snap(2, (10, 3), ["a", "b", "c"], {"a": "int64", "b": "float64", "c": "object"})
    result = compute_delta(pre, post)
    assert result["columns_added"] == ["c"]
    assert result["columns_removed"] == []

def test_columns_removed():
    pre = df_snap(1, (10, 3), ["a", "b", "c"], {"a": "int64", "b": "float64", "c": "object"})
    post = df_snap(2, (10, 2), ["a", "b"], {"a": "int64", "b": "float64"})
    result = compute_delta(pre, post)
    assert result["columns_removed"] == ["c"]
    assert result["columns_added"] == []

def test_dtype_change_detected():
    pre = df_snap(1, (10, 1), ["a"], {"a": "object"})
    post = df_snap(2, (10, 1), ["a"], {"a": "datetime64[ns]"})
    result = compute_delta(pre, post)
    assert result["dtype_changes"]["a"]["from"] == "object"
    assert result["dtype_changes"]["a"]["to"] == "datetime64[ns]"

def test_rows_delta_positive():
    pre = df_snap(1, (5, 2), ["a", "b"], {"a": "int64", "b": "int64"})
    post = df_snap(2, (10, 2), ["a", "b"], {"a": "int64", "b": "int64"})
    assert compute_delta(pre, post)["rows_delta"] == 5

def test_rows_delta_negative():
    pre = df_snap(1, (10, 2), ["a", "b"], {"a": "int64", "b": "int64"})
    post = df_snap(2, (6, 2), ["a", "b"], {"a": "int64", "b": "int64"})
    assert compute_delta(pre, post)["rows_delta"] == -4

def test_rows_delta_zero():
    pre = df_snap(1, (10, 2), ["a", "b"], {"a": "int64", "b": "int64"})
    post = df_snap(2, (10, 2), ["a", "b"], {"a": "int64", "b": "int64"})
    assert compute_delta(pre, post)["rows_delta"] == 0


# --- modification_type classification ---

def test_modification_type_addition_on_new_column():
    pre = df_snap(1, (10, 2), ["a", "b"], {"a": "int64", "b": "float64"})
    post = df_snap(2, (10, 3), ["a", "b", "c"], {"a": "int64", "b": "float64", "c": "object"})
    assert compute_delta(pre, post)["modification_type"] == "addition"

def test_modification_type_removal_on_dropped_column():
    pre = df_snap(1, (10, 3), ["a", "b", "c"], {"a": "int64", "b": "float64", "c": "object"})
    post = df_snap(2, (10, 2), ["a", "b"], {"a": "int64", "b": "float64"})
    assert compute_delta(pre, post)["modification_type"] == "removal"

def test_modification_type_removal_on_row_filter():
    pre = df_snap(1, (10, 2), ["a", "b"], {"a": "int64", "b": "float64"})
    post = df_snap(2, (6, 2), ["a", "b"], {"a": "int64", "b": "float64"})
    assert compute_delta(pre, post)["modification_type"] == "removal"

def test_modification_type_addition_on_row_append():
    pre = df_snap(1, (5, 2), ["a", "b"], {"a": "int64", "b": "float64"})
    post = df_snap(2, (10, 2), ["a", "b"], {"a": "int64", "b": "float64"})
    assert compute_delta(pre, post)["modification_type"] == "addition"

def test_modification_type_casting_on_dtype_change():
    pre = df_snap(1, (10, 1), ["a"], {"a": "object"})
    post = df_snap(2, (10, 1), ["a"], {"a": "int64"})
    assert compute_delta(pre, post)["modification_type"] == "casting"


# --- array delta ---

def test_array_delta_shape_change():
    pre = arr_snap(1, (10, 3), "float32")
    post = arr_snap(2, (10, 4), "float32")
    result = compute_delta(pre, post)
    assert result["kind"] == "array"
    assert result["shape_before"] == (10, 3)
    assert result["shape_after"] == (10, 4)

def test_array_delta_dtype_change():
    pre = arr_snap(1, (10, 3), "float32")
    post = arr_snap(2, (10, 3), "float64")
    result = compute_delta(pre, post)
    assert result["dtype_before"] == "float32"
    assert result["dtype_after"] == "float64"


# --- collection delta ---

def test_collection_delta_growth():
    result = compute_delta(col_snap(1, 5), col_snap(2, 8))
    assert result["kind"] == "collection"
    assert result["length_delta"] == 3

def test_collection_delta_shrink():
    result = compute_delta(col_snap(1, 10), col_snap(2, 4))
    assert result["length_delta"] == -6


# --- cross-kind fallback ---

def test_cross_kind_returns_generic():
    pre = df_snap(1, (5, 2), ["a", "b"], {"a": "int64", "b": "float64"})
    post = col_snap(2, 3)
    assert compute_delta(pre, post)["kind"] == "generic"
