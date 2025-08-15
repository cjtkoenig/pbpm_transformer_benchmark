import pandas as pandas_lib

def generate_prefixes(
    event_log_dataframe,
    end_of_case_token="<eoc>",
    max_prefix_length=None,
    attribute_mode="minimal"
):
    """Generate prefixes from an event log for predictive tasks."""
    prefix_records = []
    next_activity_labels = []

    for case_identifier, case_group in event_log_dataframe.groupby("case:concept:name"):
        activity_sequence = case_group["concept:name"].tolist()
        timestamp_sequence = case_group["time:timestamp"].tolist()

        for prefix_length in range(1, len(activity_sequence) + 1):
            if max_prefix_length and prefix_length > max_prefix_length:
                continue

            prefix_sequence = activity_sequence[:prefix_length]
            next_activity = (
                activity_sequence[prefix_length]
                if prefix_length < len(activity_sequence)
                else end_of_case_token
            )

            prefix_records.append({
                "case_id": case_identifier,
                "prefix_activities": prefix_sequence,
                "prefix_timestamps": timestamp_sequence[:prefix_length],
                "prefix_length": prefix_length
            })
            next_activity_labels.append(next_activity)

    prefixes_dataframe = pandas_lib.DataFrame(prefix_records)
    label_series = pandas_lib.Series(next_activity_labels, name="next_activity")
    return prefixes_dataframe, label_series
