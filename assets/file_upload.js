(function () {
  function byId(id) {
    return document.getElementById(id);
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

  function getUploadInput() {
    const wrapper = byId("file-upload");
    if (!wrapper) {
      return null;
    }
    return wrapper.querySelector('input[type="file"]');
  }

  function isRootMode() {
    return (byId("upload-directory-mode")?.textContent || "subfolder").trim() === "root";
  }

  function applyUploadMode() {
    const input = getUploadInput();
    if (!input) {
      return;
    }

    if (isRootMode()) {
      input.setAttribute("webkitdirectory", "");
      input.setAttribute("directory", "");
      input.setAttribute("mozdirectory", "");
      input.webkitdirectory = true;
      input.directory = true;
      input.mozdirectory = true;
    } else {
      input.removeAttribute("webkitdirectory");
      input.removeAttribute("directory");
      input.removeAttribute("mozdirectory");
      input.webkitdirectory = false;
      input.directory = false;
      input.mozdirectory = false;
    }
  }

  function captureRelativePaths() {
    const input = getUploadInput();
    if (!input) {
      return;
    }

    const files = Array.from(input.files || []);
    const relativePaths = files.map((file) => file.webkitRelativePath || file.name);
    setDashValue("file-upload-relative-paths", JSON.stringify(relativePaths));
  }

  function syncFilesToDashInput(files) {
    const input = getUploadInput();
    if (!input || !window.DataTransfer) {
      return;
    }

    const transfer = new DataTransfer();
    files.forEach((file) => transfer.items.add(file));
    input.files = transfer.files;
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function openFolderPicker() {
    const picker = document.createElement("input");
    picker.type = "file";
    picker.multiple = true;
    picker.setAttribute("webkitdirectory", "");
    picker.setAttribute("directory", "");
    picker.setAttribute("mozdirectory", "");
    picker.webkitdirectory = true;
    picker.directory = true;
    picker.mozdirectory = true;
    picker.style.display = "none";

    picker.addEventListener("change", function () {
      const files = Array.from(picker.files || []);
      const relativePaths = files.map((file) => file.webkitRelativePath || file.name);
      setDashValue("file-upload-relative-paths", JSON.stringify(relativePaths));
      syncFilesToDashInput(files);
      picker.remove();
    });

    document.body.appendChild(picker);
    picker.click();
  }

  function handleUploadClick(event) {
    if (!isRootMode()) {
      return;
    }

    event.preventDefault();
    event.stopPropagation();
    openFolderPicker();
  }

  function attachUploadHelpers() {
    const wrapper = byId("file-upload");
    const input = getUploadInput();
    if (!wrapper || !input) {
      return;
    }

    applyUploadMode();

    if (wrapper.dataset.folderPickerAttached !== "true") {
      wrapper.dataset.folderPickerAttached = "true";
      wrapper.addEventListener("click", handleUploadClick, true);
    }

    if (input.dataset.folderUploadAttached === "true") {
      return;
    }

    input.dataset.folderUploadAttached = "true";
    input.addEventListener("change", captureRelativePaths);
  }

  document.addEventListener("DOMContentLoaded", function () {
    attachUploadHelpers();
    const observer = new MutationObserver(function () {
      applyUploadMode();
      attachUploadHelpers();
    });
    observer.observe(document.body, { childList: true, subtree: true, characterData: true });
  });
})();
