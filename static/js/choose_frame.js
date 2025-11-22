// patient_frame_logic.js
// Final integrated version with device-aware camera handling
// Requires: jQuery, Select2

(function () {
  "use strict";

  let stopBtn = null;   // declare globally, outside document.ready
  let capturing = false; // also declare capturing globally


  $(document).ready(function () {
    // -------------------------
    // Cached nodes
    // -------------------------
    const $patientSelect = $('#patient-select');
    const page1 = document.getElementById('page-1');
    const page2 = document.getElementById('page-2');
    const btnNext = document.getElementById('btn-next');
    const btnBack = document.getElementById('btn-back');
    const btnMobileBack = document.getElementById('mobile-back');
    const liveCamera = document.getElementById('live-camera');               // desktop/Pi inline feed
    const browserCamera = document.getElementById('browser-camera');         // new <video> for mobile/tablet
    const cameraOverlay = document.getElementById('cameraOverlay');
    const overlayCameraFeed = document.getElementById('overlayCameraFeed');
    const overlayClose = document.getElementById('overlayClose');
    const overlayCloseBottom = document.getElementById('overlayCloseBottom');
    const overlayStartCapture = document.getElementById('overlayStartCapture');
    const imageContainer = document.getElementById('image-container');
    const statusBox = document.getElementById('patient-status');
    const frameShapeFilter = document.getElementById('frame-shape-filter');
    const shapeButtonsRow = document.getElementById('shape-buttons-row');
    const shapeButtons = frameShapeFilter ? frameShapeFilter.querySelectorAll('.shape-btn') : [];
    const recommendedContainer = document.getElementById('recommended-frames-container');
    const startButton = document.getElementById('start-capture');

    const overlayToggle = document.getElementById('overlay-toggle');
    const mobileOverlay = document.getElementById('recommended-overlay');

    overlayToggle.addEventListener('click', () => {
      if (mobileOverlay.classList.contains('open')) {
        // start closing animation
        mobileOverlay.classList.remove('open');
        mobileOverlay.classList.add('closing');
        overlayToggle.innerHTML =
          '<i class="fas fa-glasses"></i><span class="toggle-label font-weight-bold" style="color: #9C2627;">Recommended Frames</span>';

        mobileOverlay.addEventListener(
          'animationend',
          () => {
            mobileOverlay.classList.remove('closing');
            // stays hidden via base CSS (opacity 0, transform down)
          },
          { once: true }
        );
      } else {
        mobileOverlay.classList.add('open');
        overlayToggle.innerHTML =
          '<i class="fas fa-chevron-down"></i><span class="toggle-label font-weight-bold" style="color: #9C2627;">Close</span>';
      }
    });

    const instructionsToggle = document.getElementById('instructions-toggle');
    const instructionsOverlay = document.getElementById('instructions-overlay');

    instructionsToggle.addEventListener('click', () => {
      if (instructionsOverlay.classList.contains('open')) {
        instructionsOverlay.classList.remove('open');
        instructionsOverlay.classList.add('closing');
        instructionsToggle.innerHTML =
          '<i class="fas fa-info-circle"></i><span class="toggle-label font-weight-bold" style="color: #9C2627;">Instructions</span>';

        instructionsOverlay.addEventListener(
          'animationend',
          () => {
            instructionsOverlay.classList.remove('closing');
          },
          { once: true }
        );
      } else {
        instructionsOverlay.classList.add('open');
        instructionsToggle.innerHTML =
          '<i class="fas fa-chevron-left"></i><span class="toggle-label font-weight-bold" style="color: #9C2627;">Close</span>';
      }
    });


    if (!btnNext || !btnBack || !$patientSelect.length || !startButton) {
      console.warn('Essential DOM elements missing - patient_frame_logic.js aborted.');
      return;
    }

    const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

    // -------------------------
    // Utilities
    // -------------------------
    const videoFeedURL = liveCamera ? (liveCamera.dataset.feedUrl || '') : '';

    function isMobileDevice() {
      return /Mobi|Android/i.test(navigator.userAgent);
    }
    function isTabletDevice() {
      return /Tablet|iPad/i.test(navigator.userAgent);
    }

    function startBrowserCamera() {
      if (!browserCamera) return;
      navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
          browserCamera.srcObject = stream;
          browserCamera.style.display = 'block';

          const canvas = document.getElementById('camera-overlay-canvas');
          const ctx = canvas.getContext('2d');

          // Load Haar cascade once
          let faceCascade;
          cv['onRuntimeInitialized'] = () => {
            faceCascade = new cv.CascadeClassifier();
            faceCascade.load('/static/cascades/haarcascade_frontalface_default.xml');
          };

          function drawOverlay() {
            if (!browserCamera.videoWidth) {
              requestAnimationFrame(drawOverlay);
              return;
            }
            canvas.width = browserCamera.videoWidth;
            canvas.height = browserCamera.videoHeight;
            canvas.style.width = browserCamera.clientWidth + "px";
            canvas.style.height = browserCamera.clientHeight + "px";


            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            const axisX = 110, axisY = 140;

            // Draw oval guide (default red)
            ctx.beginPath();
            ctx.ellipse(centerX, centerY, axisX, axisY, 0, 0, 2 * Math.PI);
            ctx.strokeStyle = 'red';
            ctx.lineWidth = 2;
            ctx.stroke();

            // Rectangle guide
            const rectW = 200, rectH = 200;
            ctx.strokeStyle = 'gray';
            ctx.strokeRect(centerX - rectW/2, centerY - rectH/2, rectW, rectH);

            // Center dot
            ctx.beginPath();
            ctx.arc(centerX, centerY, 5, 0, 2 * Math.PI);
            ctx.fillStyle = 'blue';
            ctx.fill();

            // Face detection
            if (faceCascade) {
              let src = cv.imread("camera-overlay-canvas");
              let gray = new cv.Mat();
              cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
              let faces = new cv.RectVector();
              faceCascade.detectMultiScale(src, faces, 1.1, 3, 0);

              if (faces.size() > 0) {
                let face = faces.get(0);
                const faceCx = face.x + face.width/2;
                const faceCy = face.y + face.height/2;

                // Check if inside oval
                const ellipseEq = ((faceCx - centerX)**2)/(axisX**2) + ((faceCy - centerY)**2)/(axisY**2);
                if (ellipseEq <= 1.0) {
                  // Change oval to green if aligned
                  ctx.beginPath();
                  ctx.ellipse(centerX, centerY, axisX, axisY, 0, 0, 2 * Math.PI);
                  ctx.strokeStyle = 'green';
                  ctx.lineWidth = 2;
                  ctx.stroke();
                }

                ctx.fillStyle = "white";
                ctx.font = "22px Arial";
                ctx.fillText("Please align your face inside the oval", centerX - 180, centerY + axisY + 40);


                // Draw face rectangle + center dot
                ctx.strokeStyle = 'lightgray';
                ctx.strokeRect(face.x, face.y, face.width, face.height);
                ctx.beginPath();
                ctx.arc(faceCx, faceCy, 3, 0, 2 * Math.PI);
                ctx.fillStyle = 'cyan';
                ctx.fill();
              }

              src.delete(); faces.delete();
            }

            requestAnimationFrame(drawOverlay);
          }
          drawOverlay();
        })
        .catch(err => console.error("Browser camera failed:", err));
    }


    function stopBrowserCamera() {
      if (!browserCamera || !browserCamera.srcObject) return;
      let tracks = browserCamera.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      browserCamera.srcObject = null;
      browserCamera.style.display = 'none';
    }

    function addThumbnail(data) {
      if (!data || !data.img_id || !data.image_url) return;

      let thumbContainer = document.getElementById('mobile-thumb-container');
      if (!thumbContainer) {
        thumbContainer = document.createElement('div');
        thumbContainer.id = 'mobile-thumb-container';
        Object.assign(thumbContainer.style, {
          position: 'fixed',
          top: '10px',
          right: '10px',
          zIndex: '2147483647',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '8px',
          pointerEvents: 'auto',
          background: 'rgba(255,255,255,0.7)',
          padding: '10px',
          borderRadius: '6px'
        });
        document.body.appendChild(thumbContainer);

        // 🔹 Create inner scrollable wrapper
        const scrollWrapper = document.createElement('div');
        scrollWrapper.id = 'thumb-scroll-wrapper';
        Object.assign(scrollWrapper.style, {
          maxHeight: '300px',
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '8px',
          width: '100%'
        });
        thumbContainer.appendChild(scrollWrapper);
      }

      const scrollWrapper = document.getElementById('thumb-scroll-wrapper');

      // build thumbnail wrapper
      const wrapper = document.createElement('div');
      wrapper.className = 'thumb-wrapper';

      const imgWrapper = document.createElement('div');
      imgWrapper.className = 'thumb-image-wrapper';
      imgWrapper.style.position = 'relative';
      imgWrapper.style.display = 'inline-block';

      const img = document.createElement('img');
      img.src = data.image_url;
      img.alt = `Frame ${data.img_id}`;
      Object.assign(img.style, {
        maxWidth: '100px',
        borderRadius: '6px',
        boxShadow: '0 4px 12px rgba(0,0,0,0.3)'
      });
      img.title = `Frame ID ${data.img_id}`;

      const delBtn = document.createElement('button');
      delBtn.className = 'btn btn-sm btn-danger delete-thumb position-absolute';
      delBtn.style.top = '5px';
      delBtn.style.right = '5px';
      delBtn.innerHTML = '<i class="fas fa-trash"></i>';
      delBtn.onclick = async function (e) {
        e.stopPropagation();
        const file = img.getAttribute('data-filename');
        if (!confirm('Are you sure you want to delete this image?')) return;
        try {
          const fd = new FormData();
          fd.append('filename', file);
          const r = await fetch('/delete_photo', { method: 'POST', body: fd });
          if (r.ok) {
            wrapper.classList.add('fade-out');
            setTimeout(() => wrapper.remove(), 400);
          } else {
            alert('❌ Failed to delete image.');
          }
        } catch (err) {
          console.error('delete error', err);
          alert('❌ Error deleting image.');
        }
      };

      imgWrapper.appendChild(img);
      imgWrapper.appendChild(delBtn);
      wrapper.appendChild(imgWrapper);

      scrollWrapper.appendChild(wrapper);

      requestAnimationFrame(() => {
        wrapper.classList.add('show');
      });
    }


    // Global variable to hold the last FormData
    let lastFormData = null;


    // captureAndUploadMobile returns a Promise and uses the passed patientId (no shadowing)
    function captureAndUploadMobile(patientId) {
      return new Promise((resolve, reject) => {
        if (!browserCamera) {
          return resolve({ error: true, message: 'No camera' });
        }

        const canvas = document.createElement('canvas');
        canvas.width = browserCamera.videoWidth || 640;
        canvas.height = browserCamera.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(browserCamera, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(blob => {
          if (!blob) return reject({ error: true, message: 'Capture failed' });

          const formData = new FormData();
          formData.append('patient_id', patientId);
          formData.append('file', blob, 'capture.jpg');

          // ✅ Save for later use in analyze
          lastFormData = formData;

          fetch('/take_photo', {
            method: 'POST',
            body: formData
          })
          .then(resp => resp.json())
          .then(data => {
            resolve(data);
          })
          .catch(err => {
            console.error("Upload failed:", err);
            reject({ error: true, message: 'Upload failed' });
          });
        }, 'image/jpeg');
      });
    }


    // -------------------------
    // Page navigation (Next / Back)
    // -------------------------
    btnNext.addEventListener('click', function () {
      page1.classList.remove('active');
      page2.classList.add('active');

      if (window.innerWidth > 992) {
        // Desktop → use Flask stream
        if (videoFeedURL && liveCamera) {
          liveCamera.src = videoFeedURL;
          liveCamera.style.display = 'inline-block';
        }
      } else {
        // Mobile/Tablet → fullscreen camera
        document.getElementById('mobile-camera-container').classList.remove('d-none');
        startBrowserCamera();
      }
    });

   // ---------- Replace this block (startButton handler / mobile capture trigger / mobile upload) ----------

    // wire main Start button (desktop) - overlayStartCapture will call this too by calling startButton.click()
    async function startHandler(e) {
      if (e && typeof e.preventDefault === 'function') { e.preventDefault(); e.stopPropagation(); }
      if (capturing) return;
      capturing = true;

      // ensure a patient selected
      const patientId = ($patientSelect.val() || '').toString();
      if (!patientId) {
        alert("⚠️ Please select a patient before capturing images.");
        capturing = false;
        return;
      }

      if (window.innerWidth <= 992) {
          const mobileContainer = document.getElementById('mobile-camera-container');
          mobileContainer.classList.remove('d-none');
          mobileContainer.style.display = 'flex';
          startBrowserCamera();
      } else {
          if (videoFeedURL && liveCamera) {
            liveCamera.src = videoFeedURL;
            liveCamera.style.display = 'inline-block';
          }
      }

      if (!stopBtn) {
          stopBtn = document.createElement('button');
          stopBtn.id = 'stop-capture1';
          stopBtn.className = 'btn btn-success ml-1';
          stopBtn.innerHTML = '<i class="fas fa-stop mr-2"></i> Stop Capture';

          stopBtn.addEventListener('click', () => {
            capturing = false;

            const thumbContainer = document.getElementById('mobile-thumb-container');
            const scrollWrapper = document.getElementById('thumb-scroll-wrapper');

            if (thumbContainer) {
              // Check if Analyze button already exists
              if (!document.getElementById('analyze-btn')) {
                const analyzeBtn = document.createElement('button');
                analyzeBtn.id = 'analyze-btn';
                analyzeBtn.className = 'btn btn-success btn-sm analyze-btn';
                analyzeBtn.textContent = 'Analyze Frame';

                analyzeBtn.onclick = async function () {
                  if (!lastFormData) {
                    alert('❌ No captured frame available to analyze.');
                    return;
                  }

                  analyzeBtn.disabled = true;
                  analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i> Analyzing...';

                  try {
                    const r = await fetch('/analyze', {
                      method: 'POST',
                      body: lastFormData
                    });

                    if (r.redirected) {
                      window.location.href = r.url;
                    } else {
                      const html = await r.text();
                      document.open();
                      document.write(html);
                      document.close();
                    }
                  } catch (err) {
                    console.error(err);
                    alert('❌ Error analyzing frame.');
                  }
                };

                // ✅ Append analyze button to the container, after scroll wrapper
                thumbContainer.appendChild(analyzeBtn);
              }
            }
      });

      // Place Stop button in correct row
      if (window.innerWidth <= 992) {
            const btnRow = document.getElementById('mobile-button-row');
            if (btnRow) {
              btnRow.appendChild(stopBtn);
            }
          } else {
            if (startButton && startButton.parentNode) {
              startButton.parentNode.insertBefore(stopBtn, startButton.nextSibling);
            }
          }
      }

      // The loop uses overlays recreated every iteration, attached to active camera element's parent
      let countdownEl = null, progressEl = null, errorEl = null, successEl = null, invalidEl = null;

      async function doCountdown(seconds, label = 'Capturing...') {
        if (!countdownEl || !progressEl) return;
        showEl(countdownEl);
        showEl(progressEl);

        // prepare container styling for a visually large centered display
        countdownEl.style.display = 'flex';
        countdownEl.style.flexDirection = 'column';
        countdownEl.style.alignItems = 'center';
        countdownEl.style.justifyContent = 'center';
        countdownEl.style.gap = '10px';
        countdownEl.style.padding = '12px 18px';
        countdownEl.style.borderRadius = '12px';

        const isOverlayFeed = countdownEl.parentNode && countdownEl.parentNode.contains(overlayCameraFeed);

        for (let i = seconds; i > 0; i--) {
          if (isOverlayFeed) {
            countdownEl.innerHTML = `<div class="pulse-number" style="font-size:3.2rem; line-height:1; font-weight:900; transition: transform 180ms ease;">${i}</div>
                                     <div class="count-label" style="font-size:1.2rem; margin-top:6px; color:#fff; font-weight:700;">${label}</div>`;
          } else {
            countdownEl.innerHTML = `<div class="pulse-number" style="font-size:6.2rem; line-height:1; font-weight:900; transition: transform 180ms ease;">${i}</div>
                                     <div class="count-label" style="font-size:1.6rem; margin-top:6px; color:#fff; font-weight:700;">${label}</div>`;
          }

          const numEl = countdownEl.querySelector('.pulse-number');
          if (numEl) {
            numEl.style.transform = 'scale(1.45)';
            await sleep(220);
            numEl.style.transform = 'scale(1)';
            await sleep(780);
          } else {
            await sleep(1000);
          }

          if (!capturing) return;
        }

        // show camera emoji and "Captured!"
        if (isOverlayFeed) {
          countdownEl.innerHTML = `<div class="camera-emoji" style="font-size:3.6rem; line-height:1;"><i class="fas fa-camera"></i></div>
                                   <div class="count-label" style="font-size:1.2rem; margin-top:6px; color:#fff; font-weight:700;">Captured!</div>`;
        } else {
          countdownEl.innerHTML = `<div class="camera-emoji" style="font-size:5.6rem; line-height:1;"><i class="fas fa-camera"></i></div>
                                   <div class="count-label" style="font-size:1.6rem; margin-top:6px; color:#fff; font-weight:700;">Captured!</div>`;
        }

        const emojiEl = countdownEl.querySelector('.camera-emoji');
        if (emojiEl) {
          for (let k = 0; k < 4; k++) {
            emojiEl.style.transform = 'scale(1.6)';
            await sleep(180);
            emojiEl.style.transform = 'scale(1.0)';
            await sleep(180);
          }
        }

        await sleep(800);
        countdownEl.remove();
        countdownEl = null;
        await sleep(400);
      }

      // HTTP capture call (desktop or mobile)
      async function captureOnce(patientId) {
        if (window.innerWidth > 992) {
          // Desktop/Pi → GET
          try {
            const res = await fetch(`/take_photo?patient_id=${encodeURIComponent(patientId)}`);
            return await res.json();
          } catch (err) {
            console.error('captureOnce error', err);
            return { error: true, message: 'Network error' };
          }
        } else {
          // Mobile/Tablet → POST with blob (handled by captureAndUploadMobile now)
          return await captureAndUploadMobile(patientId);
        }
      }

      // main loop
      while (capturing) {
        const activeCam = getActiveCameraElement();

        // create overlays attached to active camera's parent
        countdownEl = createOverlay(activeCam, 'countdown', '');
        progressEl = createOverlay(activeCam, 'progress', '');
        errorEl = createOverlay(activeCam, 'error', '❌ ArUco marker not detected. Please adjust and retake.');
        successEl = createOverlay(activeCam, 'success', '✅ Captured successfully!');
        invalidEl = createOverlay(activeCam, 'invalid', '');

        // ensure progress label (we'll update inside doCountdown)
        progressEl.textContent = '';

        // Run countdown (5 seconds)
        await doCountdown(5, 'Capturing...');
        if (!capturing) break;

        let data;
        if (window.innerWidth <= 992) {
          data = await captureAndUploadMobile(patientId); // await here
        } else {
          data = await captureOnce(patientId);
        }


        // handle responses
        if (data && data.aruco_error) {
          showEl(errorEl);
          await sleep(1800);
          hideEl(errorEl);
          continue; // try again
        }

        if (data && data.aruco_invalid) {
          invalidEl.textContent = `❌ Frame ID ${data.detected_id} is NOT part of the recommended frame list. Please choose another one from the list.`;
          showEl(invalidEl);
          await sleep(2000);
          hideEl(invalidEl);
          continue;
        }

        // success feedback
        if (data && data.message) {
          showEl(successEl);
          await sleep(1100);
          hideEl(successEl);
        }

        // append captured image
        if (data && data.image_url) {
          imageContainer.appendChild(createCapturedImage(data));
        }

        // ✅ add thumbnail for mobile flow
        if (window.innerWidth <= 992 && data && data.img_id  && data.image_url) {
          addThumbnail(data);
        }

        // brief pause before next capture
        await sleep(350);
      } // end while

      // cleanup UI
      if (countdownEl) countdownEl.innerHTML = '🛑 Stopped';
      setTimeout(() => {
        [countdownEl, progressEl, errorEl, successEl, invalidEl].forEach(el => el && el.remove());
        if (stopBtn) { stopBtn.remove(); stopBtn = null; }
        // hide overlay feed on mobile if used (only hide when capture loop ended)
        if (window.innerWidth <= 992) hidePseudoOverlay();
      }, 700);

      capturing = false;
    } // end startHandler


    // Capture button handler (mobile overlay) — call startHandler directly to avoid click bubbling issues
    const mobileCaptureBtn = document.getElementById('mobile-start-capture');
    if (mobileCaptureBtn) {
      mobileCaptureBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        const patientId = $('#patient-select').val();
        if (!patientId) {
          alert("⚠️ Please select a patient before capturing images.");
          return;
        }
        // call start handler directly (no DOM click)
        startHandler(e);
      });
    }

    btnBack.addEventListener('click', function () {
      page2.classList.remove('active');
      page1.classList.add('active');

      if (liveCamera) {
        liveCamera.src = '';
        liveCamera.style.display = 'none';
      }
      stopBrowserCamera();
      hidePseudoOverlay();
    });

    // -------------------------
    // Select2 init
    // -------------------------
    $patientSelect.select2({ placeholder: "-- Choose Patient --", allowClear: true, width: 'resolve' });

    // -------------------------
    // Overlay helpers
    // -------------------------
    function showPseudoOverlay() {
      if (!cameraOverlay) return;
      cameraOverlay.classList.remove('d-none');
      cameraOverlay.style.position = 'fixed';
      cameraOverlay.style.display = 'flex';
      cameraOverlay.style.justifyContent = 'center';
      cameraOverlay.style.alignItems = 'center';
      cameraOverlay.style.width = '100%';
      cameraOverlay.style.height = '100%';
      cameraOverlay.style.zIndex = '9999';
      cameraOverlay.style.overflow = 'hidden';
    }



    function hidePseudoOverlay() {
      if (!cameraOverlay) return;
      cameraOverlay.classList.add('d-none');
      cameraOverlay.style.display = 'none';
    }

    if (overlayClose) {
      overlayClose.addEventListener('click', function (e) {
        e.stopPropagation();
        hidePseudoOverlay();
        stopBrowserCamera();
      });
    }

    if (overlayCloseBottom) {
      overlayCloseBottom.addEventListener('click', function (e) {
        e.preventDefault();
        e.stopPropagation();
        hidePseudoOverlay();
        stopBrowserCamera();
        if (liveCamera) liveCamera.src = '';
      });
    }

    if (overlayStartCapture) {
      overlayStartCapture.addEventListener('click', function (e) {
        e.preventDefault();
        e.stopPropagation();
        startButton.click();
      });
    }

    // -------------------------
    // Patient status & frame rendering (unchanged from your code)
    // -------------------------
    $patientSelect.on('change', async function () {
      const patientId = $(this).val();
      const recommendedContainer = document.getElementById('recommended-frames-container');
      const mobileOverlay = document.getElementById('recommended-overlay');
      recommendedContainer.innerHTML = '';
      $('#next-button-container').hide();
      if (frameShapeFilter) frameShapeFilter.style.display = 'none';

      // Reset first
        statusBox.className = "";
        statusBox.innerHTML = "";

        // Show checking animation
        statusBox.className = "badge badge-success checking-badge";
        statusBox.innerHTML = `
          <span class="spinner-border spinner-border-sm mr-2" role="status" aria-hidden="true"></span>
          Checking...
        `;


      try {
        const resp = await fetch('/check_patient_status', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ patient_id: patientId })
        });
        const data = await resp.json();
        if (data.error) {
          statusBox.innerHTML = `<span class="badge badge-warning">${data.error}</span>`;
          return;
        }

        // Build modern UI layout
        let conditionTitle = '';
        let conditionIcon = '';
        if (data.is_high_prescription) {
          conditionIcon = '<i class="fas fa-glasses mr-2" style="color: #f6c23e;"></i>';
          conditionTitle = "High Prescription";
        } else if (data.is_acidic) {
          conditionIcon = '<i class="fas fa-flask mr-2" style="color: #e74a3b;"></i>';
          conditionTitle = "Acidic Condition";
        } else if (data.is_child) {
          conditionIcon = '<i class="fas fa-baby mr-2" style="color: #36b9cc;"></i>';
          conditionTitle = "Child / Active Lifestyle";
        } else {
          conditionIcon = '<i class="fas fa-check mr-2" style="color: #1cc88a;"></i>';
          conditionTitle = "Normal";
        }

        statusBox.className = "modern-status-box";
                statusBox.innerHTML = `
          <div class="condition-title">${conditionIcon}${conditionTitle}</div>
          <div class="recommend-row">
              <span class="badge badge-material"><i class="fas fa-layer-group mr-1"></i>${data.recommended_material || "N/A"}</span>
              <span class="badge badge-frame"><i class="fas fa-glasses mr-1"></i>${data.recommended_frame || "N/A"}</span>
          </div>
          ${data.recommendation_reason ? `<div class="recommend-description">${data.recommendation_reason}</div>` : ''}
        `;

        // Show Next button now
        $('#next-button-container').fadeIn(250);

        // Render frames
        const frames = data.recommended_frames || [];
        if (frames.length === 0) {
          recommendedContainer.innerHTML = `<div class="text-muted">No frames matched your filters.</div>`;
          mobileOverlay.innerHTML = `<div class="text-muted">No frames matched your filters.</div>`;
          return;
        }

        const requiredShapes = ["Round","Oval","Rectangle","Cat eye","Geometric","Circle"];
        const grouped = {};
        requiredShapes.forEach(s => grouped[s] = []);
        frames.forEach(f => {
          const shape = (f.shape || 'Other').trim();
          if (grouped[shape]) grouped[shape].push(f);
        });

        // clear old buttons
        const shapeButtonsRow = document.getElementById('shape-buttons-row');
        const frameShapeFilter = document.getElementById('frame-shape-filter');
        shapeButtonsRow.innerHTML = '';
        frameShapeFilter.innerHTML = '';

        // helper to create buttons
        function createShapeButton(label, group) {
          const btn = document.createElement('button');
          btn.className = 'btn btn-secondary m-1 shape-btn';
          btn.textContent = label;
          btn.dataset.shape = label;
          btn.onclick = function () {
            document.querySelectorAll('.shape-btn').forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            renderRecommendedFrames(group);
          };
          return btn;
        }

        // ✅ All button shows all frames
        const allBtn = createShapeButton("All", frames);
        const allBtn2 = allBtn.cloneNode(true);
        allBtn2.onclick = allBtn.onclick;

        // mark All as active by default
        allBtn.classList.add('active');
        allBtn2.classList.add('active');

        // append to both desktop and overlay
        frameShapeFilter.appendChild(allBtn);
        shapeButtonsRow.appendChild(allBtn2);

        // render all frames initially
        renderRecommendedFrames(frames);

        // build shape buttons
        requiredShapes.forEach(shape => {
          if (grouped[shape].length > 0) {
            const btn = createShapeButton(shape, grouped[shape]);
            const btn2 = btn.cloneNode(true);
            btn2.onclick = btn.onclick;
            frameShapeFilter.appendChild(btn);
            shapeButtonsRow.appendChild(btn2);
          }
        });

        // always show filters
        frameShapeFilter.style.display = 'block';
      } catch (err) {
        console.error('check_patient_status error', err);
        statusBox.innerHTML = `<span class="badge badge-warning">⚠ Error connecting to server</span>`;
      }
    });

    // -------------------------
    // Frame card + modal
    // -------------------------
    function showFrameModal(imageUrl, frameInfo = '', frameBadge = '') {
      let overlay = document.getElementById('frame-modal-overlay');
      if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'frame-modal-overlay';
        Object.assign(overlay.style, {
          position: 'fixed',
          top: '0',
          left: '0',
          width: '100%',
          height: '100%',
          backgroundColor: 'rgba(0,0,0,0.7)',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          zIndex: '99999',   // 🔥 raise above camera feed
          cursor: 'pointer',
          opacity: '0',
          transition: 'opacity 0.3s ease',
          padding: '10px'
        });

        document.body.appendChild(overlay);
        overlay.addEventListener('click', () => {
          overlay.style.opacity = '0';
          setTimeout(() => { overlay.style.display = 'none'; }, 300);
        });
      }
      overlay.innerHTML = '';
      const imgWrapper = document.createElement('div');
      Object.assign(imgWrapper.style, {
        position: 'relative',
        width: '100%',
        maxWidth: '450px',
        maxHeight: '90vh',
        textAlign: 'center',
        transform: 'scale(0.8)',
        transition: 'transform 0.3s ease',
        cursor: 'default'
      });
      const img = document.createElement('img');
      img.src = imageUrl;
      Object.assign(img.style, {
        width: '100%',
        height: 'auto',
        borderRadius: '12px',
        boxShadow: '0 8px 30px rgba(0,0,0,0.5)',
        objectFit: 'contain'
      });
      imgWrapper.appendChild(img);
      const infoContainer = document.createElement('div');
      Object.assign(infoContainer.style, {
        marginTop: '10px',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '6px'
      });
      const infoText = document.createElement('div');
      infoText.textContent = frameInfo;
      Object.assign(infoText.style, {
        color: '#fff',
        fontSize: '1rem',
        fontWeight: '500',
        textAlign: 'center'
      });
      infoContainer.appendChild(infoText);
      if (frameBadge) {
        const badge = document.createElement('span');
        badge.textContent = frameBadge;
        badge.className = 'badge badge-primary';
        Object.assign(badge.style, {
          fontSize: '0.85rem',
          padding: '3px 7px'
        });
        infoContainer.appendChild(badge);
      }
      imgWrapper.appendChild(infoContainer);
      overlay.appendChild(imgWrapper);
      overlay.style.display = 'flex';
      setTimeout(() => {
        overlay.style.opacity = '1';
        imgWrapper.style.transform = 'scale(1)';
      }, 20);
    }

    function createFrameCard(frame, mode = 'desktop') {
      const imgUrl = frame.image_url || '/static/img/default-frame.png';
      const frameInfo = `${frame.brand || 'Unknown'} ${frame.model_number || ''}`;
      const frameBadge = frame.material || '';

      if (mode === 'desktop') {
        const c = document.createElement('div');
        c.className = 'card m-2 p-2 text-center';
        c.style.width = '200px';
        c.innerHTML = `
          <img src="${imgUrl}" class="img-fluid rounded mb-2 selectable-frame" style="height:120px; object-fit:contain; cursor:pointer;">
          <div style="font-size:0.9rem;">
            <strong>${frame.brand || 'Unknown'}</strong><br>
            <small>${frame.model_number || ''}</small><br>
            <small>${frame.color || ''} • ${frame.shape || ''}</small><br>
            <small><em>${frame.material || ''} | ${frame.frame_type || ''}</em></small><br>
            <span class="text-success font-weight-bold">₱${(Number(frame.price)||0).toFixed(2)}</span>
          </div>
        `;
        const imgEl = c.querySelector('.selectable-frame');
        imgEl.addEventListener('click', () => {
          showFrameModal(imgUrl, frameInfo, frameBadge);
        });
        return c;
      } else {
          const card = document.createElement('div');
          card.className = 'frame-card';
          card.style.background = 'rgba(255,255,255,0.7)';
          card.innerHTML = `
            <img src="${imgUrl}" class="img-fluid rounded mb-2"
                 style="height:100px; object-fit:contain; cursor:pointer;">
            <div style="font-size:0.8rem;">
              <strong>${frame.brand || 'Unknown'}</strong><br>
              <small>${frame.model_number || ''}</small><br>
              <small>${frame.color || ''} • ${frame.shape || ''}</small><br>
              <small><em>${frame.material || ''} | ${frame.frame_type || ''}</em></small><br>
              <span class="mb-2 text-success font-weight-bold">
                ₱${(Number(frame.price)||0).toFixed(2)}
              </span>
            </div>
          `;
          // ✅ Attach handler to the whole card
          card.addEventListener('click', () => {
            showFrameModal(imgUrl, frameInfo, frameBadge);
          });
          return card;
      }
    }

    function renderRecommendedFrames(frames) {
      const desktopContainer = document.getElementById('recommended-frames-container');
      const overlayCards = document.getElementById('overlay-frame-cards');

      desktopContainer.innerHTML = '';
      overlayCards.innerHTML = '';

      if (!frames || frames.length === 0) {
        overlayCards.innerHTML = `<div class="text-muted">No frames matched your filters.</div>`;
        return;
      }

      frames.forEach(f => {
        desktopContainer.appendChild(createFrameCard(f, 'desktop'));
        overlayCards.appendChild(createFrameCard(f, 'mobile'));
      });

      // show overlay
      document.getElementById('recommended-overlay').classList.add('open');
    }


    // Attach overlay to parent of camera element or to cameraOverlay content area.
    function createOverlay(attachedToElement, type = 'info', defaultText = '') {
      // If no element passed, fallback to browserCamera or liveCamera
      if (!attachedToElement) {
        attachedToElement = window.innerWidth <= 992
          ? document.getElementById('browser-camera')
          : document.body;
      }

      let container;
      if (window.innerWidth <= 992) {
        // Mobile → use wrapper around video
        container = attachedToElement;
      } else {
        // Desktop → keep using cameraOverlay
        container = attachedToElement.parentNode || attachedToElement;
      }

      // Ensure the container is positioned relative
      const cs = window.getComputedStyle(container);
      if (!cs.position || cs.position === 'static') {
        container.style.position = 'relative';
      }

      const el = document.createElement('div');
      el.className = 'camera-overlay-annotation';
      Object.assign(el.style, {
        position: 'absolute',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        padding: '14px 20px',
        borderRadius: '12px',
        fontWeight: '800',
        fontSize: '1rem',
        textAlign: 'center',
        display: 'none',
        zIndex: '2147483647',
        color: '#fff',
        maxWidth: '92%',
        wordBreak: 'break-word',
        boxShadow: '0 6px 20px rgba(0,0,0,0.45)'
      });

      switch (type) {
        case 'error':
        case 'success':
        case 'invalid': {
          const isOverlayFeed =
            attachedToElement === overlayCameraFeed ||
            (attachedToElement && attachedToElement.id === 'overlayCameraFeed');

          if (isOverlayFeed || window.innerWidth <= 992) {
            // 🔹 Mobile/tablet: shrink text and padding
            el.style.fontSize = '0.9rem';
            el.style.padding = '4px 8px';
            el.style.maxWidth = '75%';
          } else {
            // Desktop size
            el.style.fontSize = '1.2rem';
            el.style.padding = '8px 12px';
          }

          if (type === 'error') {
            el.style.background = 'rgba(220,38,38,0.92)';
          } else if (type === 'success') {
            el.style.background = 'rgba(16,185,129,0.92)';
          } else if (type === 'invalid') {
            el.style.background = 'rgba(239,68,68,0.95)';
          }
          break;
        }

        case 'countdown': {
          const isOverlayFeed =
            attachedToElement === overlayCameraFeed ||
            (attachedToElement && attachedToElement.id === 'overlayCameraFeed');

          if (isOverlayFeed || window.innerWidth <= 992) {
            // 🔹 Mobile/tablet: smaller countdown numbers/icons
            el.style.fontSize = '1.4rem';
            el.style.padding = '4px 8px';
            el.style.maxWidth = '75%';
          } else {
            el.style.fontSize = '3rem';
            el.style.padding = '10px 14px';
          }

          el.style.color = 'red';
          el.style.textShadow = '1px 1px 4px rgba(0,0,0,0.6)';
          el.style.top = '45%';
          break;
        }

        case 'progress':
          el.style.top = '90%';
          el.style.transform = 'translateX(-50%)';
          el.style.color = '#6b7280';
          el.style.fontSize = window.innerWidth <= 992 ? '0.85rem' : '1rem';
          el.style.fontWeight = '700';
          break;

        default:
          el.style.background = 'rgba(30,64,175,0.9)';
      }

      el.textContent = defaultText;
      container.appendChild(el);
      return el;
    }

    function showEl(el) { if (el) el.style.display = 'flex'; }
    function hideEl(el) { if (el) el.style.display = 'none'; }


    // -------------------------
    // Camera capture logic
    // -------------------------
    // function to determine which camera element to use (desktop inline or overlay mobile)
   function getActiveCameraElement() {
      if (window.innerWidth > 992) {
        return liveCamera;
      } else {
        return document.getElementById('mobile-camera-container');
      }
    }


    // wire main Start button (desktop) - overlayStartCapture will call this too by calling startButton.click()
    startButton.addEventListener('click', async function startHandler(e) {
      if (capturing) return;
      capturing = true;

      // ensure a patient selected
      const patientId = ($patientSelect.val() || '').toString();
      if (!patientId) {
        alert("⚠️ Please select a patient before capturing images.");
        capturing = false;
        return;
      }

      // show mobile overlay for small screens
      if (window.innerWidth <= 992) {
        showPseudoOverlay();
      } else {
        // show desktop inline feed
        if (videoFeedURL && liveCamera) {
          liveCamera.src = videoFeedURL;
          liveCamera.style.display = 'inline-block';
        }
      }

      // add stop button next to start if not already present
      if (!stopBtn) {
        stopBtn = document.createElement('button');
        stopBtn.id = 'stop-capture';
        stopBtn.className = 'btn btn-success mt-2 ml-2';
        stopBtn.innerHTML = '<i class="fas fa-stop mr-2"></i> Stop Capture';
        startButton.after(stopBtn);
        stopBtn.addEventListener('click', () => { capturing = false; });
      }

      // The loop uses overlays recreated every iteration, attached to active camera element's parent
      let countdownEl = null, progressEl = null, errorEl = null, successEl = null, invalidEl = null;

      // countdown with big pulse on numbers and camera emoji final; shows label while counting then "Captured!"
      async function doCountdown(seconds, label = 'Capturing...') {
        if (!countdownEl || !progressEl) return;
        showEl(countdownEl);
        showEl(progressEl);

        // prepare container styling for a visually large centered display
        countdownEl.style.display = 'flex';
        countdownEl.style.flexDirection = 'column';
        countdownEl.style.alignItems = 'center';
        countdownEl.style.justifyContent = 'center';
        countdownEl.style.gap = '10px';
        countdownEl.style.padding = '12px 18px';
        countdownEl.style.borderRadius = '12px';

        // detect small overlay (to adapt number sizes)
        const isOverlayFeed = countdownEl.parentNode && countdownEl.parentNode.contains(overlayCameraFeed);

        for (let i = seconds; i > 0; i--) {
          // create the number + label (size adapts for overlay)
          if (isOverlayFeed) {
            countdownEl.innerHTML = `<div class="pulse-number" style="font-size:3.2rem; line-height:1; font-weight:900; transition: transform 180ms ease;">${i}</div>
                                     <div class="count-label" style="font-size:1.2rem; margin-top:6px; color:#fff; font-weight:700;">${label}</div>`;
          } else {
            countdownEl.innerHTML = `<div class="pulse-number" style="font-size:6.2rem; line-height:1; font-weight:900; transition: transform 180ms ease;">${i}</div>
                                     <div class="count-label" style="font-size:1.6rem; margin-top:6px; color:#fff; font-weight:700;">${label}</div>`;
          }

          // pulse animation: scale up then back
          const numEl = countdownEl.querySelector('.pulse-number');
          if (numEl) {
            numEl.style.transform = 'scale(1.45)';
            await sleep(220);
            numEl.style.transform = 'scale(1)';
            await sleep(780);
          } else {
            await sleep(1000);
          }

          if (!capturing) return;
        }

        // show camera emoji and "Captured!" text (per your request: when camera icon is displayed replace the word)
        if (isOverlayFeed) {
          countdownEl.innerHTML = `<div class="camera-emoji" style="font-size:3.6rem; line-height:1;"><i class="fas fa-camera"></i></div>
                                   <div class="count-label" style="font-size:1.2rem; margin-top:6px; color:#fff; font-weight:700;">Captured!</div>`;
        } else {
          countdownEl.innerHTML = `<div class="camera-emoji" style="font-size:5.6rem; line-height:1;"><i class="fas fa-camera"></i></div>
                                   <div class="count-label" style="font-size:1.6rem; margin-top:6px; color:#fff; font-weight:700;">Captured!</div>`;
        }

        // small pulse on emoji
        const emojiEl = countdownEl.querySelector('.camera-emoji');
        if (emojiEl) {
          for (let k = 0; k < 4; k++) {
            emojiEl.style.transform = 'scale(1.6)';
            await sleep(180);
            emojiEl.style.transform = 'scale(1.0)';
            await sleep(180);
          }
        }

        // remove countdown overlay completely to avoid overlap
        await sleep(800);
        countdownEl.remove();
        countdownEl = null;

        await sleep(400);
      }

      // HTTP capture call
      async function captureOnce(patientId) {
          if (window.innerWidth > 992) {
            // Desktop/Pi → GET
            try {
              const res = await fetch(`/take_photo?patient_id=${encodeURIComponent(patientId)}`);
              return await res.json();
            } catch (err) {
              console.error('captureOnce error', err);
              return { error: true, message: 'Network error' };
            }
          } else {
            // Mobile/Tablet → POST with blob
            return new Promise((resolve, reject) => {
              const canvas = document.createElement('canvas');
              canvas.width = browserCamera.videoWidth;
              canvas.height = browserCamera.videoHeight;
              const ctx = canvas.getContext('2d');
              ctx.drawImage(browserCamera, 0, 0);

              canvas.toBlob(blob => {
                const formData = new FormData();
                formData.append('patient_id', patientId);
                formData.append('file', blob, 'capture.jpg');

                fetch('/take_photo', { method: 'POST', body: formData })
                  .then(resp => resp.json())
                  .then(data => resolve(data))
                  .catch(err => {
                    console.error("Upload failed:", err);
                    reject({ error: true, message: 'Upload failed' });
                  });
              }, 'image/jpeg');
            });
          }
      }

      // main loop
      while (capturing) {
        const activeCam = getActiveCameraElement();

        // create overlays attached to active camera's parent
        countdownEl = createOverlay(activeCam, 'countdown', '');
        progressEl = createOverlay(activeCam, 'progress', '');
        errorEl = createOverlay(activeCam, 'error', '❌ ArUco marker not detected. Please adjust and retake.');
        successEl = createOverlay(activeCam, 'success', '✅ Captured successfully!');
        invalidEl = createOverlay(activeCam, 'invalid', '');

        // ensure progress label (we'll update inside doCountdown)
        progressEl.textContent = '';

        // Run countdown (5 seconds)
        await doCountdown(5, 'Capturing...');
        if (!capturing) break;

        let data;
          if (window.innerWidth <= 992) {
            // Mobile → use browser camera upload
            captureAndUploadMobile(patientId);
            // You can set `data = { message: "Captured!", image_url: null }` if you want to reuse success feedback
          } else {
            // Desktop → use your existing backend capture
            data = await captureOnce(patientId);
          }

        // handle responses
        if (data && data.aruco_error) {
          showEl(errorEl);
          await sleep(1800);
          hideEl(errorEl);
          continue; // try again
        }

        if (data && data.aruco_invalid) {
          invalidEl.textContent = `❌ Frame ID ${data.detected_id} is NOT part of the recommended frame list. Please choose another one from the list`;
          showEl(invalidEl);
          await sleep(2000);
          hideEl(invalidEl);
          continue;
        }

        // success feedback
        if (data && data.message) {
          showEl(successEl);
          await sleep(1100);
          hideEl(successEl);
        }

        // append captured image
        if (data && data.image_url) {
          imageContainer.appendChild(createCapturedImage(data));
        }

        // brief pause before next capture
        await sleep(350);
      } // end while

      // cleanup UI
      if (countdownEl) countdownEl.innerHTML = '🛑 Stopped';
      setTimeout(() => {
        [countdownEl, progressEl, errorEl, successEl, invalidEl].forEach(el => el && el.remove());
        if (stopBtn) { stopBtn.remove(); stopBtn = null; }
        // hide overlay feed on mobile if used
        if (window.innerWidth <= 992) hidePseudoOverlay();
      }, 700);

      capturing = false;
    }); // end startButton handler

    // -------------------------
    // createCapturedImage (+ delete)
    // -------------------------
    function createCapturedImage(data) {
      const wrapper = document.createElement('div');
      wrapper.className = 'image-wrapper position-relative d-inline-block m-2';

      const img = document.createElement('img');
      img.src = data.image_url;
      img.className = 'selectable-image img-fluid rounded shadow-sm';
      img.width = 450;
      img.height = 150;
      img.setAttribute('data-filename', data.img_id || '');
      img.onclick = function () { if (typeof selectImage === 'function') selectImage(this); };

      const del = document.createElement('button');
      del.className = 'btn btn-sm btn-danger position-absolute';
      del.style.top = '5px';
      del.style.right = '5px';
      del.innerHTML = '<i class="fas fa-trash"></i>';
      del.onclick = async function (e) {
        e.stopPropagation();
        const file = img.getAttribute('data-filename');
        if (!confirm('Are you sure you want to delete this image?')) return;
        try {
          const fd = new FormData();
          fd.append('filename', file);
          const r = await fetch('/delete_photo', { method: 'POST', body: fd });
          if (r.ok) wrapper.remove();
          else alert('❌ Failed to delete image.');
        } catch (err) {
          console.error('delete error', err);
          alert('❌ Error deleting image.');
        }
      };

      wrapper.appendChild(img);
      wrapper.appendChild(del);
      return wrapper;
    }

    // -------------------------
    // Debug
    // -------------------------
    console.info('patient_frame_logic.js loaded — pseudo-overlay ready');
  });
})();
