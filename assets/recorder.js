(function () {
  let audioContext = null;
  let mediaStream = null;
  let mediaSource = null;
  let processor = null;
  let buffers = [];
  let recording = false;
  let sampleRate = 44100;

  function byId(id) {
    return document.getElementById(id);
  }

  function setStatus(text) {
    const el = byId("recording-status");
    if (el) {
      el.textContent = text;
    }
  }

  function setDashValue(id, value) {
    const el = byId(id);
    if (!el) {
      return;
    }

    const prototype =
      el.tagName === "TEXTAREA"
        ? window.HTMLTextAreaElement && window.HTMLTextAreaElement.prototype
        : window.HTMLInputElement && window.HTMLInputElement.prototype;
    const setter = prototype ? Object.getOwnPropertyDescriptor(prototype, "value")?.set : null;
    const previousValue = el.value;

    if (setter) {
      setter.call(el, value);
    } else {
      el.value = value;
    }

    if (el._valueTracker) {
      el._valueTracker.setValue(previousValue);
    }

    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function mergeBuffers(chunks) {
    let totalLength = 0;
    chunks.forEach((chunk) => {
      totalLength += chunk.length;
    });

    const result = new Float32Array(totalLength);
    let offset = 0;
    chunks.forEach((chunk) => {
      result.set(chunk, offset);
      offset += chunk.length;
    });
    return result;
  }

  function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i += 1) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  function floatTo16BitPCM(view, offset, input) {
    for (let i = 0; i < input.length; i += 1, offset += 2) {
      const sample = Math.max(-1, Math.min(1, input[i]));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    }
  }

  function encodeWav(samples, rate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    writeString(view, 0, "RIFF");
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, "WAVE");
    writeString(view, 12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, rate, true);
    view.setUint32(28, rate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, "data");
    view.setUint32(40, samples.length * 2, true);
    floatTo16BitPCM(view, 44, samples);

    return buffer;
  }

  function arrayBufferToBase64(buffer) {
    let binary = "";
    const bytes = new Uint8Array(buffer);
    const chunkSize = 0x8000;
    for (let i = 0; i < bytes.length; i += chunkSize) {
      const chunk = bytes.subarray(i, i + chunkSize);
      binary += String.fromCharCode.apply(null, chunk);
    }
    return window.btoa(binary);
  }

  async function startRecording() {
    if (recording) {
      return;
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus("This browser does not support microphone capture.");
      return;
    }

    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      sampleRate = audioContext.sampleRate;
      mediaSource = audioContext.createMediaStreamSource(mediaStream);
      processor = audioContext.createScriptProcessor(4096, 1, 1);
      buffers = [];
      recording = true;
      setDashValue("recording-data", "");

      processor.onaudioprocess = function (event) {
        if (!recording) {
          return;
        }
        const channelData = event.inputBuffer.getChannelData(0);
        buffers.push(new Float32Array(channelData));
      };

      mediaSource.connect(processor);
      processor.connect(audioContext.destination);
      const preview = byId("recording-preview-audio");
      if (preview) {
        preview.removeAttribute("src");
        preview.load();
      }
      setStatus("Recording... press Stop when finished.");
    } catch (error) {
      setStatus("Microphone access was not granted.");
    }
  }

  function cleanupRecorder() {
    if (processor) {
      processor.disconnect();
      processor.onaudioprocess = null;
      processor = null;
    }
    if (mediaSource) {
      mediaSource.disconnect();
      mediaSource = null;
    }
    if (mediaStream) {
      mediaStream.getTracks().forEach((track) => track.stop());
      mediaStream = null;
    }
    if (audioContext) {
      audioContext.close();
      audioContext = null;
    }
  }

  function stopRecording() {
    if (!recording) {
      setStatus("No active recording to stop.");
      return;
    }

    recording = false;
    const samples = mergeBuffers(buffers);
    const wavBuffer = encodeWav(samples, sampleRate);
    const dataUrl = "data:audio/wav;base64," + arrayBufferToBase64(wavBuffer);
    setDashValue("recording-data", dataUrl);

    const preview = byId("recording-preview-audio");
    if (preview) {
      preview.src = dataUrl;
      preview.load();
    }

    const seconds = samples.length / sampleRate;
    setStatus("Recording ready (" + seconds.toFixed(1) + "s). Save it when you're happy.");
    cleanupRecorder();
  }

  function attachRecorder() {
    const startButton = byId("record-start-btn");
    const stopButton = byId("record-stop-btn");

    if (!startButton || !stopButton || startButton.dataset.recorderAttached === "true") {
      return;
    }

    startButton.dataset.recorderAttached = "true";
    startButton.addEventListener("click", startRecording);
    stopButton.addEventListener("click", stopRecording);
  }

  document.addEventListener("DOMContentLoaded", function () {
    attachRecorder();
    const observer = new MutationObserver(attachRecorder);
    observer.observe(document.body, { childList: true, subtree: true });
  });
})();
