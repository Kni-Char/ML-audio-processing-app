(function () {
  window.dash_clientside = window.dash_clientside || {};
  window.dash_clientside.clientside = Object.assign({}, window.dash_clientside.clientside, {
    setProcessingStartedAt: function (n_clicks) {
      if (!n_clicks) {
        return window.dash_clientside.no_update;
      }
      return Date.now();
    },

    updateProcessingTimer: function (startedAt, n_intervals) {
      if (!startedAt) {
        return [
          "00:00",
          "Preparing pooled dataset, splitting hits, extracting features, and evaluating the selected models.",
        ];
      }

      const elapsedMs = Math.max(0, Date.now() - startedAt);
      const totalSeconds = Math.floor(elapsedMs / 1000);
      const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, "0");
      const seconds = String(totalSeconds % 60).padStart(2, "0");
      const timer = `${minutes}:${seconds}`;
      const note =
        totalSeconds < 10
          ? "Starting the processing pipeline..."
          : "Pipeline is still running. Larger pooled datasets can take a couple of minutes.";

      return [timer, note];
    },
  });
})();
