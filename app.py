from __future__ import annotations

import mimetypes

from dash import Dash
from flask import abort, send_file

from ml_audio_app.callbacks import register_callbacks
from ml_audio_app.layout import create_layout
from ml_audio_app.storage import ensure_app_directories, get_default_dataset_slug, resolve_audio_path


ensure_app_directories()
DEFAULT_ACTIVE_DATASET = get_default_dataset_slug()

app = Dash(__name__, title="ML Audio Processing App", suppress_callback_exceptions=True)
server = app.server


@server.route("/media/<dataset_slug>/<section>/<path:relative_path>")
def serve_media(dataset_slug: str, section: str, relative_path: str):
    try:
        path = resolve_audio_path(dataset_slug, section, relative_path)
    except ValueError:
        abort(404)
    if not path.exists():
        abort(404)
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return send_file(path, mimetype=mime_type, download_name=path.name, as_attachment=False)


app.layout = create_layout(DEFAULT_ACTIVE_DATASET)
register_callbacks(app, DEFAULT_ACTIVE_DATASET)


if __name__ == "__main__":
    app.run(debug=True)
