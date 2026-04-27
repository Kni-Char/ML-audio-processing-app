from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def make_waveform_figure(time_values: list[float], amplitudes: list[float], title: str, color: str) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=time_values,
            y=amplitudes,
            mode="lines",
            line={"color": color, "width": 1.5},
            hovertemplate="Time %{x:.3f}s<br>Amplitude %{y:.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
    )
    return figure


def make_onset_figure(
    onset_time_axis: list[float],
    onset_envelope: list[float],
    onset_times: list[float],
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=onset_time_axis,
            y=onset_envelope,
            mode="lines",
            line={"color": "#1d4ed8", "width": 1.5},
            name="Onset Envelope",
        )
    )

    if onset_times:
        ymax = max(onset_envelope) if onset_envelope else 1.0
        for onset in onset_times:
            figure.add_vline(x=onset, line_width=1, line_dash="dot", line_color="#dc2626")
        figure.add_trace(
            go.Scatter(
                x=onset_times,
                y=[ymax] * len(onset_times),
                mode="markers",
                marker={"color": "#dc2626", "size": 8, "symbol": "diamond"},
                name="Detected Hits",
                hovertemplate="Detected onset %{x:.3f}s<extra></extra>",
            )
        )

    figure.update_layout(
        title="Adaptive Onset Detection",
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis_title="Time (s)",
        yaxis_title="Onset Strength",
        legend={"orientation": "h"},
    )
    return figure


def make_spectrogram_figure(
    spectrogram_db: list[list[float]],
    times: list[float],
    freqs: list[float],
) -> go.Figure:
    figure = go.Figure(
        data=[
            go.Heatmap(
                z=spectrogram_db,
                x=times,
                y=freqs,
                colorscale="Viridis",
                colorbar={"title": "dB"},
                hovertemplate="Time %{x:.3f}s<br>Freq %{y:.0f}Hz<br>Level %{z:.2f}dB<extra></extra>",
            )
        ]
    )
    figure.update_layout(
        title="Processed Signal Spectrogram",
        margin={"l": 50, "r": 20, "t": 50, "b": 40},
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
    )
    return figure


def make_clip_lengths_figure(clip_lengths: list[float]) -> go.Figure:
    figure = go.Figure()
    if clip_lengths:
        figure.add_trace(
            go.Bar(
                x=list(range(1, len(clip_lengths) + 1)),
                y=clip_lengths,
                marker_color="#059669",
                hovertemplate="Hit %{x}<br>Length %{y:.3f}s<extra></extra>",
            )
        )
    figure.update_layout(
        title="Single-Hit Clip Durations",
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis_title="Detected Hit",
        yaxis_title="Duration (s)",
    )
    return figure


def make_accuracy_figure(results_rows: list[dict]) -> go.Figure:
    frame = pd.DataFrame(results_rows)
    figure = go.Figure()

    if frame.empty:
        figure.update_layout(
            title="Model Accuracy Overview",
            annotations=[
                {
                    "text": "Run the pipeline to populate model accuracy results.",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                }
            ],
        )
        return figure

    x_labels = [f"{row.feature_set} | {row.model}" for row in frame.itertuples()]
    figure.add_trace(go.Bar(name="Train", x=x_labels, y=frame["train_accuracy"], marker_color="#2563eb"))
    figure.add_trace(go.Bar(name="Validation", x=x_labels, y=frame["testing_accuracy"], marker_color="#0891b2"))
    figure.add_trace(go.Bar(name="Testing", x=x_labels, y=frame["validation_accuracy"], marker_color="#059669"))
    figure.update_layout(
        title="Model Accuracy Overview",
        barmode="group",
        margin={"l": 40, "r": 20, "t": 50, "b": 80},
        paper_bgcolor="white",
        plot_bgcolor="white",
        yaxis_title="Accuracy",
        xaxis_title="Feature Set / Model",
        xaxis={"tickangle": -20},
        legend={"orientation": "h"},
    )
    return figure


def make_confusion_matrix_figure(matrix: list[list[int]], title: str) -> go.Figure:
    labels = ["good (0)", "bad (1)"]
    figure = go.Figure(
        data=[
            go.Heatmap(
                z=matrix,
                x=labels,
                y=labels,
                colorscale="Blues",
                showscale=False,
                text=matrix,
                texttemplate="%{text}",
                hovertemplate="Predicted %{x}<br>Actual %{y}<br>Count %{z}<extra></extra>",
            )
        ]
    )
    figure.update_layout(
        title=title,
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
    )
    return figure
