from pm4py.objects.log.importer.xes import importer as xes_importer
import pandas as pandas_lib

REQUIRED_COLUMNS = ["case:concept:name", "concept:name", "time:timestamp"]

def load_event_log(file_path: str):
    """Load an event log from a .xes or .csv file into a pandas DataFrame."""
    if file_path.endswith(".xes"):
        event_log_object = xes_importer.apply(file_path)
        event_rows = []

        for trace in event_log_object:
            case_identifier = trace.attributes["concept:name"]

            for event in trace:
                event_record = {
                    "case:concept:name": case_identifier,
                    "concept:name": event["concept:name"],
                    "time:timestamp": pandas_lib.to_datetime(event["time:timestamp"])
                }
                for attribute_name, attribute_value in event.items():
                    if attribute_name not in ("concept:name", "time:timestamp"):
                        event_record[attribute_name] = attribute_value
                event_rows.append(event_record)

        return (
            pandas_lib.DataFrame(event_rows)
            .sort_values(["case:concept:name", "time:timestamp"])
            .reset_index(drop=True)
        )

    return pandas_lib.read_csv(file_path, parse_dates=["time:timestamp"])
