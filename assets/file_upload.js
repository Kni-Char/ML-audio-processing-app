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

  function getUploadMode() {
    return (byId("upload-directory-mode")?.textContent || "subfolder").trim();
  }

  function isRootMode() {
    return getUploadMode() === "root";
  }

  function isDisabledMode() {
    return getUploadMode() === "disabled";
  }

  function applyUploadMode() {
    const input = getUploadInput();
    if (!input) {
      return;
    }

    if (isDisabledMode()) {
      input.removeAttribute("webkitdirectory");
      input.removeAttribute("directory");
      input.removeAttribute("mozdirectory");
      input.webkitdirectory = false;
      input.directory = false;
      input.mozdirectory = false;
      input.disabled = true;
    } else if (isRootMode()) {
      input.setAttribute("webkitdirectory", "");
      input.setAttribute("directory", "");
      input.setAttribute("mozdirectory", "");
      input.webkitdirectory = true;
      input.directory = true;
      input.mozdirectory = true;
      input.disabled = false;
    } else {
      input.removeAttribute("webkitdirectory");
      input.removeAttribute("directory");
      input.removeAttribute("mozdirectory");
      input.webkitdirectory = false;
      input.directory = false;
      input.mozdirectory = false;
      input.disabled = false;
    }
  }

  function captureRelativePaths() {
    const input = getUploadInput();
    if (!input) {
      return;
    }

    const customRelativePaths = input.dataset.customRelativePaths;
    if (customRelativePaths) {
      setDashValue("file-upload-relative-paths", customRelativePaths);
      delete input.dataset.customRelativePaths;
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

  function collectFilesFromHandle(handle, currentPath) {
    return new Promise(function (resolve, reject) {
      if (!handle) {
        resolve([]);
        return;
      }

      if (handle.kind === "file") {
        handle
          .getFile()
          .then(function (file) {
            const relativePath = currentPath ? `${currentPath}/${file.name}` : file.name;
            resolve([{ file: file, relativePath: relativePath }]);
          })
          .catch(reject);
        return;
      }

      if (handle.kind !== "directory") {
        resolve([]);
        return;
      }

      (async function () {
        const collected = [];
        const nextPath = currentPath ? `${currentPath}/${handle.name}` : handle.name;
        for await (const child of handle.values()) {
          const childRecords = await collectFilesFromHandle(child, nextPath);
          collected.push(...childRecords);
        }
        resolve(collected);
      })().catch(reject);
    });
  }

  function syncFilesToDashInput(files, relativePaths) {
    const input = getUploadInput();
    if (!input || !window.DataTransfer) {
      return;
    }

    const transfer = new DataTransfer();
    files.forEach((file) => transfer.items.add(file));
    if (relativePaths && relativePaths.length) {
      input.dataset.customRelativePaths = JSON.stringify(relativePaths);
    } else {
      delete input.dataset.customRelativePaths;
    }
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
      syncFilesToDashInput(files, relativePaths);
      picker.remove();
    });

    document.body.appendChild(picker);
    picker.click();
  }

  function handleUploadClick(event) {
    if (isDisabledMode()) {
      event.preventDefault();
      event.stopPropagation();
      return;
    }

    if (!isRootMode()) {
      return;
    }

    event.preventDefault();
    event.stopPropagation();
    openFolderPicker();
  }

  function handleUploadDragOver(event) {
    if (isDisabledMode()) {
      event.preventDefault();
      event.stopPropagation();
      return;
    }

    event.preventDefault();
    event.stopPropagation();
    if (event.dataTransfer) {
      event.dataTransfer.dropEffect = "copy";
    }
  }

  function handleUploadDrop(event) {
    if (isDisabledMode()) {
      event.preventDefault();
      event.stopPropagation();
      return;
    }

    event.preventDefault();
    event.stopPropagation();

    if (!isRootMode()) {
      const droppedFiles = Array.from((event.dataTransfer && event.dataTransfer.files) || []);
      const relativePaths = droppedFiles.map(function (file) {
        return file.webkitRelativePath || file.name;
      });
      setDashValue("file-upload-relative-paths", JSON.stringify(relativePaths));
      syncFilesToDashInput(droppedFiles, relativePaths);
      return;
    }

    const items = Array.from((event.dataTransfer && event.dataTransfer.items) || []);

    const handlePromises = items.map(function (item) {
      if (typeof item.getAsFileSystemHandle === "function") {
        return item.getAsFileSystemHandle().catch(function () {
          return null;
        });
      }
      return Promise.resolve(null);
    });

    Promise.all(handlePromises)
      .then(function (handles) {
        const validHandles = handles.filter(Boolean);
        if (validHandles.length) {
          return Promise.all(
            validHandles.map(function (handle) {
              return collectFilesFromHandle(handle, "");
            })
          );
        }

        const entries = items
          .map(function (item) {
            return item.webkitGetAsEntry ? item.webkitGetAsEntry() : null;
          })
          .filter(Boolean);

        if (!entries.length) {
          return null;
        }

        return Promise.all(
          entries.map(function (entry) {
            return collectDroppedFiles(entry, "");
          })
        );
      })
      .then(function (results) {
        if (!results) {
          const fallbackFiles = Array.from((event.dataTransfer && event.dataTransfer.files) || []);
          const fallbackPaths = fallbackFiles.map(function (file) {
            return file.webkitRelativePath || file.name;
          });
          setDashValue("file-upload-relative-paths", JSON.stringify(fallbackPaths));
          syncFilesToDashInput(fallbackFiles, fallbackPaths);
          return;
        }

        const flattened = results.flat();
        const files = flattened.map(function (record) {
          return record.file;
        });
        const relativePaths = flattened.map(function (record) {
          return record.relativePath.replace(/^\/+/, "");
        });
        setDashValue("file-upload-relative-paths", JSON.stringify(relativePaths));
        syncFilesToDashInput(files, relativePaths);
      })
      .catch(function () {
        const fallbackFiles = Array.from((event.dataTransfer && event.dataTransfer.files) || []);
        const fallbackPaths = fallbackFiles.map(function (file) {
          return file.webkitRelativePath || file.name;
        });
        setDashValue("file-upload-relative-paths", JSON.stringify(fallbackPaths));
        syncFilesToDashInput(fallbackFiles, fallbackPaths);
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
