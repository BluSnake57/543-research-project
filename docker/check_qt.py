"""Stress the progress UI to detect native binding crashes during long downloads."""

import time

from PySide6.QtWidgets import QApplication, QTreeWidget

from sentinel_x.activity_ui import update_job_rows


def main():
    app = QApplication.instance() or QApplication([])
    table = QTreeWidget()
    table.setColumnCount(5)
    job = {
        "id": "download-stress",
        "kind": "download:qwen",
        "status": "running",
        "created": time.time(),
        "events": [],
    }
    update_job_rows(table, [job])
    item = table.topLevelItem(0)
    table.setCurrentItem(item)
    bar = table.itemWidget(item, 3)
    # At normal polling frequency this represents over an hour of UI updates.
    # PySide6 6.12.0 on Python 3.11 aborts after only hundreds of these calls.
    for index in range(5000):
        job["events"] = [
            {
                "event": "download",
                "file": "model.safetensors",
                "downloaded": index + 1,
                "total": 5000,
                "time": time.time(),
            }
        ]
        update_job_rows(table, [job])
        app.processEvents()
    assert table.currentItem() is item
    assert table.itemWidget(item, 3) is bar
    assert bar.value() == 100
    job["status"] = "completed"
    update_job_rows(table, [job])
    update_job_rows(table, [])
    assert table.topLevelItemCount() == 0
    table.close()
    print("5,000 progress refreshes, selection, completion, and removal passed.")


if __name__ == "__main__":
    main()
