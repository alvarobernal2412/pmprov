import pandas as pd


def order_classify(event):
    if event["event_count_fwd"] == 0:
        if event["event_count_bwd"] == 0:
            return "only"
        else:
            return "first"
    elif event["event_count_bwd"] == 0:
        return "last"
    else:
        return "middle"


def event_add_order_classifier(
    event_log, case_id_col="case:concept:name", time_col="time:timestamp"
):
    cases = event_log.groupby(case_id_col)
    if cases[time_col].is_monotonic_increasing.all():
        event_log["event_count_fwd"] = cases.cumcount()
        event_log["event_count_bwd"] = cases["event_count_fwd"].transform(
            lambda x: x.max() - x
        )
        event_log["order_class"] = event_log.apply(order_classify, axis=1)
        event_log = event_log.drop(columns=["event_count_fwd", "event_count_bwd"])
    else:
        print("cases are not ordered in time")
    return event_log


def create_case_log(event_log):
    """
    creates an initial case log
    """
    cases = event_log.groupby("case:concept:name")
    case_log = cases.agg(
        start_time=("time:timestamp", "first"),
        end_time=("time:timestamp", "last"),
        no_of_events=("concept:name", "count"),
    )
    case_log["duration"] = case_log["end_time"] - case_log["start_time"]
    return case_log


def log_track_status_count(
    event_log,
    status_col="concept:name",
    status_abstraction=(lambda x: x),
    case_id_col="case:concept:name",
    time_col="time:timestamp",
):
    """
    runs through all events in order of their timestamp and computes how many cases are in which state,
    adding counting columns to the event log
    states are distinguished from a given column status_col of the event log, which can be mapped through a status_abstraction
    if the status_abstraction maps to a null value, the state is unchanged
    """
    assert event_log[
        time_col
    ].is_monotonic_increasing, (
        "The event log time stamps are not monotonically increasing."
    )

    status_list = [
        status_abstraction(status)
        for status in event_log[status_col].unique()
        if (status_abstraction(status) is not None)
    ]
    status_list = list(set(status_list))  # remove duplicates
    case_id_list = list(event_log[case_id_col].unique())

    value_store = {status: 0 for status in status_list}
    INITIAL_STATUS = "init"
    status_list.append(INITIAL_STATUS)
    value_store[INITIAL_STATUS] = len(case_id_list)
    added_cols = {status: [] for status in status_list}

    case_status = {case_id: INITIAL_STATUS for case_id in case_id_list}
    for index, row in event_log.iterrows():
        new_status = status_abstraction(row[status_col])
        if new_status is not None:
            this_case_id = row[case_id_col]
            old_status = case_status[this_case_id]
            value_store[old_status] -= 1
            case_status[this_case_id] = new_status
            value_store[new_status] += 1
        for status in status_list:
            added_cols[status].append(value_store[status])
    for status in status_list:
        col_name = status + "_count"
        event_log[col_name] = added_cols[status]

    return event_log


def track_variable(log, column_name, tracking_method, case_id_col="case:concept:name"):
    return track_variables(
        log, [(column_name, tracking_method)], case_id_col=case_id_col
    )


def init_value_of_tracking_method(tracking_method):
    match tracking_method:
        case "latest":
            return None
        case "count":
            return 0
        case "sum":
            return 0
        case "min":
            return 2000000.0
        case "max":
            return 0
        case "append":
            return list()
        case _:
            print("initialization:", tracking_method, "NOT SUPPORTED")


def track_variables(
    log, tracking_list, case_id_col="case:concept:name", time_col="time:timestamp"
):
    """
    tracks the state of a variable, i.e., column, of an event along the evolution of each case,
    and records the track in a separate column;
    tracking logic is defined by the tracking method
    multiple variables can be tracked, specified in tracking_list
    """
    assert log.groupby(case_id_col)[
        time_col
    ].is_monotonic_increasing.all(), (
        "events of each case must be sorted in the order of their timestamp"
    )

    value_store = (
        {}
    )  # map each tracked variable (incl method) and case to its current value
    added_cols = (
        {}
    )  # map each tracked variable (incl method) to a growing list/vector of values
    added_col_names = []

    for col, method in tracking_list:
        added_cols[(col, method)] = []

    for index, row in log.iterrows():
        case_id = row[case_id_col]
        for col, method in tracking_list:
            if (case_id, col, method) not in value_store:
                value_store[(case_id, col, method)] = init_value_of_tracking_method(
                    method
                )
            if not pd.isna(row[col]):
                match method:
                    case "latest":
                        value_store[(case_id, col, method)] = row[col]
                    case "count":
                        value_store[(case_id, col, method)] = (
                            value_store[(case_id, col, method)] + 1
                        )
                    case "max":
                        value_store[(case_id, col, method)] = max(
                            row[col], value_store[(case_id, col, method)]
                        )
                    case "min":
                        value_store[(case_id, col, method)] = min(
                            row[col], value_store[(case_id, col, method)]
                        )
                    case "sum":
                        value_store[(case_id, col, method)] = round(
                            value_store[(case_id, col, method)] + row[col], 2
                        )
                    case "append":
                        value_store[(case_id, col, method)].append(row[col])
                    case _:
                        print("update", method, "NOT SUPPORTED")
            added_cols[(col, method)].append(value_store[(case_id, col, method)])

    for col, method in tracking_list:
        tracking_col_name = col + "::" + method
        added_col_names.append(tracking_col_name)
        log[tracking_col_name] = added_cols[(col, method)]

    return log


def enrich_with_row_map(log, column_name, mapper):
    log[column_name] = log.apply(mapper, axis=1)
    return log


def compute_sorted_case_id(event, case_log, case_id_col="case:concept:name"):
    """
    prefixes the case id with the start time of the case such that ordering by
    the resulting sorted case id will order cases according to their arrival time
    """
    case_start_time = case_log.at[event[case_id_col], "start_time"]
    return str(case_start_time) + "_" + event[case_id_col]


def event_add_ordered_case_id(event_log, case_log, case_id_col="case:concept:name"):
    """
    enriches the event log with a sorted case id
    """
    event_log["ordered_case_id"] = event_log.apply(
        compute_sorted_case_id, case_log=case_log, axis=1
    )
    return event_log


def event_add_relative_case_time(
    event_log, case_id_col="case:concept:name", time_col="time:timestamp"
):
    """
    adds the relative case time for each event in the log
    the earliest event of each case has relative time 0
    """
    event_log["rel_time"] = event_log.groupby(case_id_col)[time_col].transform(
        lambda x: x - x.min()
    )
    return event_log


def event_add_relative_log_time(event_log, time_col="time:timestamp"):
    """
    adds the relative log time for each event in the log
    the earliest event of the log has relative time 0
    """
    event_log["rel_log_time"] = event_log[time_col] - event_log[time_col].min()
    return event_log


def convert_name(name):
    return "_".join(name.split(" "))


def case_add_activity_start_time(
    case_log, event_log, activity, case_id_col="case:concept:name", time_col="rel_time"
):
    """
    enrich a case log by adding the start time of first occurrence of a given activity in each case
    """
    filtered_event_log = event_log[event_log["concept:name"] == activity]
    col_name = convert_name(activity) + "::start"
    case_log[col_name] = filtered_event_log.groupby(case_id_col)[time_col].first()
    return case_log


def case_add_activity_start_times(
    case_log,
    event_log,
    case_id_col="case:concept:name",
    activity_col="concept:name",
    time_col="rel_time",
):
    """
    enrich a case log by adding, for each activity, the start time of it's first occurrence in each case
    """
    cases = event_log.groupby([case_id_col, activity_col])
    activity_starts = cases[time_col].first()
    activities = event_log[activity_col].unique()
    for activity in activities:
        col_name = convert_name(activity) + "::start"
        case_log[col_name] = activity_starts.xs(activity, level=1, axis=0)
    return case_log


def case_add_activity_incidence(
    case_log, event_log, case_id_col="case:concept:name", activity_col="concept:name"
):
    """
    enriches a case log by adding for each activity, the number of occurrences in a case
    """
    cases = event_log.groupby(case_id_col)
    activity_incidence = cases[activity_col].value_counts()
    activities = event_log[activity_col].unique()
    for activity in activities:
        col_name = convert_name(activity) + "::count"
        case_log[col_name] = activity_incidence.xs(activity, level=1, axis=0)
        case_log[col_name] = case_log[col_name].fillna(0)
    return case_log


def case_add_activity_delay(case_log, activity1, activity2):
    """
    enrich a case log by adding, for a given pair of activities, the delay between their first occurrences in each case
    requires a 'start' time to be present for each activity in the case log
    """
    activity_name1 = convert_name(activity1)
    activity_name2 = convert_name(activity2)
    col_name = activity_name1 + ":" + activity_name2 + "::delay"
    case_log[col_name] = (
        case_log[activity_name2 + "::start"] - case_log[activity_name1 + "::start"]
    )
    return case_log


def case_add_activity_delays(case_log, event_log, activity_col="concept:name"):
    """
    enrich a case log by adding, for each unordered pair of activities, the delay between their first occurrences in each case
    """
    activities = list(event_log[activity_col].unique())
    first_activities = activities.copy()
    while len(first_activities) > 0:
        first_act = first_activities.pop()
        second_activities = first_activities.copy()
        while len(second_activities) > 0:
            second_act = second_activities.pop()
            case_log = case_add_activity_delay(case_log, first_act, second_act)
    return case_log


def case_add_activity_attributes(
    case_log,
    event_log,
    attributes=None,
    case_id_col="case:concept:name",
    activity_col="concept:name",
):
    """
    Enriches a case log by adding specified data attributes from each occurrence of each activity.
    For activities that occur multiple times in a case, creates numbered columns (e.g., Activity:2.attribute).

    Parameters:
    - attributes: list of column names to add. If None, adds all columns except case_id, activity, and timestamp columns
    """
    if attributes is None:
        # Get all columns except the core event log columns
        excluded_cols = [
            case_id_col,
            activity_col,
            "time:timestamp",
            "concept:name",
            "case:concept:name",
        ]
        attributes = [col for col in event_log.columns if col not in excluded_cols]

    # Track occurrence count for each activity in each case
    activity_occurrence = {}

    # Sort by case and maintain order
    event_log_sorted = (
        event_log.sort_values([case_id_col, "time:timestamp"])
        if "time:timestamp" in event_log.columns
        else event_log
    )

    for index, row in event_log_sorted.iterrows():
        case_id = row[case_id_col]
        activity = row[activity_col]

        # Track which occurrence this is for this activity in this case
        key = (case_id, activity)
        if key not in activity_occurrence:
            activity_occurrence[key] = 1
            qualifier = convert_name(activity) + "."
        else:
            activity_occurrence[key] += 1
            qualifier = (
                convert_name(activity) + ":" + str(activity_occurrence[key]) + "."
            )

        # Add each attribute with the qualifier
        for attr in attributes:
            if attr in row and not pd.isna(row[attr]):
                col_name = qualifier + attr
                case_log.at[case_id, col_name] = row[attr]

    return case_log


def brute_force_case_log_enricher(
    case_log,
    event_log,
    include_time=True,
    include_attributes=True,
    attributes=None,
    case_id_col="case:concept:name",
    activity_col="concept:name",
):
    """
    enriches a case log by adding for each activity, the number of occurrences in a case, as well as start times, delays, and data attributes

    Parameters:
    - include_attributes: if True, adds data attributes from events to case log
    - attributes: list of attribute columns to include. If None and include_attributes=True, includes all available attributes
    """
    case_log = case_add_activity_incidence(
        case_log, event_log, case_id_col=case_id_col, activity_col=activity_col
    )
    if include_time:
        case_log = case_add_activity_start_times(
            case_log, event_log, case_id_col=case_id_col, activity_col=activity_col
        )
        case_log = case_add_activity_delays(
            case_log, event_log, activity_col=activity_col
        )
    if include_attributes:
        case_log = case_add_activity_attributes(
            case_log,
            event_log,
            attributes=attributes,
            case_id_col=case_id_col,
            activity_col=activity_col,
        )
    return case_log
