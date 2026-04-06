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

  function readAllEntries(reader) {
    return new Promise(function (resolve, reject) {
      const entries = [];

      function readBatch() {
        reader.readEntries(
          function (batch) {
            if (!batch.length) {
              resolve(entries);
              return;
            }
            entries.push(...batch);
            readBatch();
          },
          function (error) {
            reject(error);
          }
        );
      }

      readBatch();
    });
  }

  function collectDroppedFiles(entry, currentPath) {
    return new Promise(function (resolve, reject) {
      if (entry.isFile) {
        entry.file(
          function (file) {
            const relativePath = currentPath ? `${currentPath}/${file.name}` : file.name;
            resolve([{ file: file, relativePath: relativePath }]);
          },
          function (error) {
            reject(error);
          }
        );
        return;
      }

      if (!entry.isDirectory) {
        resolve([]);
        return;
      }

      readAllEntries(entry.createReader())
        .then(function (children) {
          return Promise.all(
            children.map(function (child) {
              const childPath = currentPath ? `${currentPath}/${entry.name}` : entry.name;
              return collectDroppedFiles(child, childPath);
            })
          );
        })
        .then(function (results) {
          resolve(results.flat());
        })
        .catch(reject);
    });
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

  function handleUploadDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
    if (event.dataTransfer) {
      event.dataTransfer.dropEffect = "copy";
    }
  }

  function handleUploadDrop(event) {
    event.preventDefault();
    event.stopPropagation();

    if (!isRootMode()) {
      const droppedFiles = Array.from((event.dataTransfer && event.dataTransfer.files) || []);
      const relativePaths = droppedFiles.map(function (file) {
        return file.webkitRelativePath || file.name;
      });
      setDashValue("file-upload-relative-paths", JSON.stringify(relativePaths));
      syncFilesToDashInput(droppedFiles);
      return;
    }

    const items = Array.from((event.dataTransfer && event.dataTransfer.items) || []);
    const entries = items
      .map(function (item) {
        return item.webkitGetAsEntry ? item.webkitGetAsEntry() : null;
      })
      .filter(Boolean);

    if (!entries.length) {
      const fallbackFiles = Array.from((event.dataTransfer && event.dataTransfer.files) || []);
      const fallbackPaths = fallbackFiles.map(function (file) {
        return file.webkitRelativePath || file.name;
      });
      setDashValue("file-upload-relative-paths", JSON.stringify(fallbackPaths));
      syncFilesToDashInput(fallbackFiles);
      return;
    }

    Promise.all(
      entries.map(function (entry) {
        return collectDroppedFiles(entry, "");
      })
    )
      .then(function (results) {
        const flattened = results.flat();
        const files = flattened.map(function (record) {
          return record.file;
        });
        const relativePaths = flattened.map(function (record) {
          return record.relativePath.replace(/^\/+/, "");
        });
        setDashValue("file-upload-relative-paths", JSON.stringify(relativePaths));
        syncFilesToDashInput(files);
      })
      .catch(function () {
        const fallbackFiles = Array.from((event.dataTransfer && event.dataTransfer.files) || []);
        const fallbackPaths = fallbackFiles.map(function (file) {
          return file.webkitRelativePath || file.name;
        });
        setDashValue("file-upload-relative-paths", JSON.stringify(fallbackPaths));
        syncFilesToDashInput(fallbackFiles);
      });
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
      wrapper.addEventListener("dragover", handleUploadDragOver, true);
      wrapper.addEventListener("drop", handleUploadDrop, true);
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
