% estimate_formants_pitch_cepstral.m
% Usage: run in folder containing 'hindi_uu.wav' (or change wavfile below)
clear; close all; clc;

%% Parameters
wavfile = 'hindi_uu.wav';    % change if needed
preemph = 0.97;
frame_dur = 0.03;            % 30 ms frames
hop_dur   = 0.01;            % 10 ms hop
lifter_ms = 1.5;             % low-time lifter cutoff in milliseconds (controls smoothing)
max_formant_freq = 5000;     % Hz (search for peaks up to this freq)
min_pitch = 50;              % Hz
max_pitch = 500;             % Hz
peak_prominence = 0.1;       % adapt if needed for peak detection (relative)
min_consecutive_voiced = 6;  % requirement from user

%% Read audio
[x, fs] = audioread(wavfile);
if size(x,2) > 1, x = mean(x,2); end
x = x / max(abs(x));  % normalize

% Pre-emphasis
x = filter([1 -preemph], 1, x);

%% Framing
frame_len = round(frame_dur * fs);
hop_len   = round(hop_dur * fs);
nfft = 2^nextpow2(frame_len*4); % zero-pad for smoother spectra
win = hamming(frame_len);

frames = buffer(x, frame_len, frame_len - hop_len, 'nodelay');
numFrames = size(frames,2);

% Remove last column if shorter than frame_len (buffer with nodelay may truncate)
if size(frames,1) < frame_len
    frames = frames(1:frame_len,:);
end

% Time axis for frame centers
frame_times = ((0:(numFrames-1))*hop_len + frame_len/2) / fs;

%% Pre-allocate
cepstra = zeros(nfft, numFrames);      % complex ifft length nfft
smoothedLogSpec = zeros(nfft/2+1, numFrames);
freqAxis = (0:(nfft/2))*(fs/nfft);
queffAxis = (0:(nfft-1))/fs;           % seconds (quefrency)

% Lifter index (convert lifter_ms to samples/time domain)
lifter_n = round(lifter_ms * 1e-3 * fs); % number of samples to keep around 0 quefrency

pitchHz = zeros(1, numFrames);
F = nan(3, numFrames); % F(1,:) F1, F(2,:) F2, F(3,:) F3

%% Framewise processing
for k = 1:numFrames
    fr = frames(:,k).*win;
    X = fft(fr, nfft);
    magX = abs(X(1:nfft/2+1));
    logMag = log(magX + eps);
    
    % real cepstrum (via IFFT of log magnitude)
    c = ifft([logMag; flipud(logMag(2:end-1))]); % reconstruct full spectrum for real ifft
    c = real(c); % length nfft
    
    cepstra(:,k) = c;
    
    % lifter: low-time liftering -> smooth the log spectrum
    % zero out high quefrency components (keep low quefrency components)
    cl = c;
    % keep indices 1 : lifter_n and last lifter_n (mirror)
    if lifter_n < length(c)/2
        cl(lifter_n+1 : end-lifter_n) = 0;
    end
    % back to (smoothed) log spectrum
    smLogFull = fft(cl);
    smLogPos = real(smLogFull(1:nfft/2+1)); % positive freqs
    smoothedLogSpec(:,k) = smLogPos;
    
    % formant estimation from smoothed log magnitude:
    % find peaks in the smoothed log magnitude up to max_formant_freq
    fIdxMax = find(freqAxis <= max_formant_freq, 1, 'last');
    [pks, locs] = findpeaks(smLogPos(1:fIdxMax), 'MinPeakProminence', peak_prominence);
    % if not enough peaks, relax prominence
    if length(pks) < 3
        [pks, locs] = findpeaks(smLogPos(1:fIdxMax));
    end
    % sort peaks by frequency (locs already in ascending freq), choose the three largest peaks by amplitude
    if ~isempty(pks)
        [~, sortidx] = sort(pks, 'descend');
        topN = min(3, length(pks));
        chosen = sortidx(1:topN);
        chosen_locs = locs(chosen);
        chosen_freqs = freqAxis(chosen_locs);
        % sort chosen freqs ascending to map to F1<F2<F3 order
        chosen_freqs = sort(chosen_freqs);
        % assign to F matrix
        F(1:length(chosen_freqs),k) = chosen_freqs(:);
    end
    
    % Pitch estimation from cepstrum:
    % Search cepstrum for peak in quefrency range corresponding to [max_pitch .. min_pitch]
    qmin = round(fs / max_pitch);  % smallest quefrency index (in samples)
    qmax = round(fs / min_pitch);  % largest quefrency index (in samples)
    qmin = max(qmin,2); qmax = min(qmax, floor(nfft/2)); % avoid index 1 (0 quefrency)
    cepSlice = c(qmin:qmax);
    if ~isempty(cepSlice)
        [mp, mi] = max(cepSlice);
        peakIndex = (mi - 1) + qmin; % index into c
        pitch_est = fs / peakIndex;  % Hz
        pitchHz(k) = pitch_est;
    else
        pitchHz(k) = 0;
    end
end

%% Voicing decision: use cepstral peak magnitude (normalized) and energy
cep_peak_mag = zeros(1,numFrames);
frame_energy = sum(frames.^2);
for k = 1:numFrames
    % use cepstrum magnitude in the pitch search region as voicing metric
    qmin = round(fs / max_pitch);
    qmax = round(fs / min_pitch);
    qmin = max(qmin,2); qmax = min(qmax, floor(nfft/2));
    c = cepstra(:,k);
    if qmax >= qmin
        cep_peak_mag(k) = max(c(qmin:qmax));
    else
        cep_peak_mag(k) = 0;
    end
end
% normalize metrics
cep_peak_mag = cep_peak_mag - min(cep_peak_mag);
if max(cep_peak_mag)>0, cep_peak_mag = cep_peak_mag / max(cep_peak_mag); end
frame_energy = frame_energy - min(frame_energy);
if max(frame_energy)>0, frame_energy = frame_energy / max(frame_energy); end

voiced_mask = (cep_peak_mag > 0.25) & (frame_energy > 0.1); % thresholds; may change per-file

% enforce at least 6 consecutive frames: find longest run of voiced frames
runs = bwlabel(voiced_mask);
run_ids = unique(runs(runs>0));
best_run_len = 0; best_run_id = 0;
for rid = run_ids'
    lenr = sum(runs==rid);
    if lenr > best_run_len
        best_run_len = lenr;
        best_run_id = rid;
    end
end

if best_run_len >= min_consecutive_voiced
    voiced_frames_idx = find(runs==best_run_id);
else
    % fallback: take central voiced frames by energy (top min_consecutive_voiced frames)
    [~, idxsort] = sort(frame_energy, 'descend');
    voiced_frames_idx = sort(idxsort(1:min(min_consecutive_voiced, numFrames)));
    warning('Could not find %d contiguous voiced frames; using %d highest-energy frames as fallback.', ...
            min_consecutive_voiced, length(voiced_frames_idx));
end

%% Compute averages over the chosen frames (only where F exists)
sel = voiced_frames_idx;
F1_frames = F(1,sel); F2_frames = F(2,sel); F3_frames = F(3,sel);
% remove NaNs
F1_mean = mean(F1_frames(~isnan(F1_frames)));
F2_mean = mean(F2_frames(~isnan(F2_frames)));
F3_mean = mean(F3_frames(~isnan(F3_frames)));
pitch_mean = mean(pitchHz(sel(pitchHz(sel)>0)));

% Print summary
fprintf('\nSelected frames (indices): %s\n', mat2str(sel));
fprintf('Average F1 over selected frames: %.1f Hz\n', F1_mean);
fprintf('Average F2 over selected frames: %.1f Hz\n', F2_mean);
fprintf('Average F3 over selected frames: %.1f Hz\n', F3_mean);
fprintf('Average pitch (f0) over selected frames: %.1f Hz\n\n', pitch_mean);

%% Plots
% 1) Cepstrally smoothed spectra (image)
figure('Name','Cepstrally smoothed log-spectrum (frames vs frequency)');
imagesc(frame_times, freqAxis, smoothedLogSpec);
axis xy;
xlabel('Time (s)'); ylabel('Frequency (Hz)');
title('Cepstrally-smoothed log spectrum (frames)');
colorbar;
ylim([0 5000]);

% 2) Cepstral sequence (quefrency vs time)
figure('Name','Cepstral sequence (frames vs quefrency)');
% show only quefrency up to e.g. 20 ms (important for pitch & envelope)
max_quef_ms = 20;
max_quef_idx = min(nfft, round(max_quef_ms*1e-3*fs));
imagesc(frame_times, (0:max_quef_idx-1)/fs*1000, cepstra(1:max_quef_idx,:));
axis xy;
xlabel('Time (s)'); ylabel('Quefrency (ms)');
title('Cepstral sequence (low quefrency region)');
colorbar;

% 3) Framewise pitch contour & voiced mask
figure('Name','Pitch contour and voicing');
subplot(2,1,1);
plot(frame_times, pitchHz, '-o'); hold on;
plot(frame_times(sel), pitchHz(sel), 'ro','MarkerFaceColor','r');
xlabel('Time (s)'); ylabel('Pitch (Hz)');
title('Framewise pitch (cepstral peak method)');
ylim([0 max(500, max(pitchHz)+20)]);
grid on;
subplot(2,1,2);
plot(frame_times, cep_peak_mag, '-');
hold on; plot(frame_times, frame_energy, '--'); legend('Cepstral peak mag (norm)','Energy (norm)');
hold on; stem(frame_times, voiced_mask, 'k');
xlabel('Time (s)'); ylabel('Normalized');
title('Voicing decision metrics and mask');

% 4) Formants over time (F1-F3)
figure('Name','Formant tracks (from smoothed spectrum)');
plot(frame_times, F(1,:), 'b.-'); hold on;
plot(frame_times, F(2,:), 'g.-');
plot(frame_times, F(3,:), 'r.-');
plot(frame_times(sel), F(1,sel), 'bo','MarkerFaceColor','b');
xlabel('Time (s)'); ylabel('Frequency (Hz)');
legend('F1','F2','F3','Location','northwest');
title('Estimated formant tracks from cepstrally-smoothed spectrum');
ylim([0 5000]);
grid on;

%% Save results to struct (optional)
results.Ftracks = F;
results.pitchPerFrame = pitchHz;
results.smoothedLogSpec = smoothedLogSpec;
results.cepstra = cepstra;
results.frame_times = frame_times;
results.selected_frames = sel;
results.avgF1 = F1_mean;
results.avgF2 = F2_mean;
results.avgF3 = F3_mean;
results.avgPitch = pitch_mean;

% save('formant_pitch_results.mat','results');
