 
clear; close all; clc;

%% PARAMETERS
wav_path = 'hindi_dha.wav';  % change to your filename
preemph = 0.97;
frame_dur = 0.03;         % 30 ms
hop_dur = 0.01;           % 10 ms
lifter_ms = 1.5;          % ms lifter for smoothing
max_formant_freq = 5000;  % Hz
min_pitch = 50;           % Hz
max_pitch = 500;          % Hz
peak_prominence = 0.1;
min_consecutive_voiced = 6;

%% READ AUDIO
[x, fs] = audioread(wav_path);
if size(x,2) > 1
    x = mean(x,2); % stereo → mono
end
if all(x==0)
    error('Audio is silent.');
end
x = x ./ max(abs(x)); % normalize

% Pre-emphasis
x = [x(1); x(2:end) - preemph * x(1:end-1)];

%% FRAMING
frame_len = round(frame_dur * fs);
hop_len = round(hop_dur * fs);
if length(x) < frame_len
    error('Audio too short for chosen frame duration.');
end
num_frames = 1 + floor((length(x) - frame_len) / hop_len);

frames = zeros(frame_len, num_frames);
for i = 1:num_frames
    start_idx = (i-1)*hop_len + 1;
    frames(:,i) = x(start_idx:start_idx+frame_len-1);
end

win = hamming(frame_len);
frame_times = ((0:num_frames-1)*hop_len + frame_len/2) / fs;

% FFT size (4× zero-padding)
nfft = 1;
while nfft < frame_len * 4
    nfft = nfft * 2;
end
M = nfft/2 + 1;
freq_axis = linspace(0, fs/2, M);

% Lifter in samples
lifter_n = round(lifter_ms * 1e-3 * fs);

cepstra = zeros(nfft, num_frames);
smoothed_log_spec = zeros(M, num_frames);
pitchHz = zeros(1, num_frames);
F = nan(3, num_frames);

%% FRAMEWISE PROCESSING
for k = 1:num_frames
    fr = frames(:,k) .* win;
    X = fft(fr, nfft);
    magX = abs(X(1:M)) + eps;
    logMag = log(magX);
    
    % Real cepstrum
    mirrored = [logMag; logMag(end-1:-1:2)];
    c = real(ifft(mirrored));
    cepstra(:,k) = c;
    
    % Liftering
    cl = c;
    if lifter_n < length(c)/2
        cl(lifter_n+1:end-lifter_n) = 0;
    end
    
    % Smoothed log spectrum
    smLogFull = fft(cl);
    smLogPos = real(smLogFull(1:M));
    smoothed_log_spec(:,k) = smLogPos;
    
    % ---------- FORMANT ESTIMATION ----------
    fidx_max = find(freq_axis <= max_formant_freq, 1, 'last');
    if isempty(fidx_max)
        fidx_max = M;
    end
    % ensure nonnegative and small floor
    rel_prom = max(0.001, peak_prominence * max(0, max(smLogPos(1:fidx_max))));
    [pks, locs] = findpeaks(smLogPos(1:fidx_max), 'MinPeakProminence', rel_prom);
    if isempty(pks)
        [pks, locs] = findpeaks(smLogPos(1:fidx_max));
    end
    if ~isempty(pks)
        [~, idx_sort] = sort(pks, 'descend');
        topN = min(3, length(idx_sort));
        chosen_freqs = sort(freq_axis(locs(idx_sort(1:topN))));
        F(1:topN, k) = chosen_freqs;
    end
    
    % ---------- PITCH ESTIMATION ----------
    qmin = round(fs / max_pitch);
    qmax = round(fs / min_pitch);
    qmin = max(qmin, 2);
    qmax = min(qmax, floor(nfft/2));
    if qmax >= qmin && qmin <= length(c)
        qmax_clamped = min(qmax, length(c));
        cep_slice = c(qmin:qmax_clamped);
        if ~isempty(cep_slice)
            [~, mi] = max(cep_slice);
            peak_idx = qmin + mi - 1;
            if peak_idx > 0
                pitch_est = fs / peak_idx;
                pitchHz(k) = pitch_est;
            end
        end
    end
end

%% VOICING DECISION
cep_peak_mag = zeros(1, num_frames);
frame_energy = sum(frames.^2, 1);
for k = 1:num_frames
    qmin = round(fs / max_pitch);
    qmax = round(fs / min_pitch);
    qmin = max(qmin, 2);
    qmax = min(qmax, floor(nfft/2));
    c = cepstra(:,k);
    if qmax >= qmin && qmin <= length(c)
        qmax_clamped = min(qmax, length(c));
        cep_peak_mag(k) = max(c(qmin:qmax_clamped));
    else
        cep_peak_mag(k) = 0;
    end
end

% Normalize (explicit max-min to avoid using range)
denom1 = max(cep_peak_mag) - min(cep_peak_mag);
if denom1 == 0, denom1 = 1e-8; end
cep_peak_mag = (cep_peak_mag - min(cep_peak_mag)) ./ (denom1 + 1e-12);

denom2 = max(frame_energy) - min(frame_energy);
if denom2 == 0, denom2 = 1e-8; end
frame_energy_norm = (frame_energy - min(frame_energy)) ./ (denom2 + 1e-12);

voiced_mask = (cep_peak_mag > 0.25) & (frame_energy_norm > 0.1);

% Find consecutive voiced frames
runs = zeros(1, num_frames);
run_id = 0;
i = 1;
while i <= num_frames
    if voiced_mask(i)
        run_id = run_id + 1;
        j = i;
        while j <= num_frames && voiced_mask(j)
            runs(j) = run_id;
            j = j + 1;
        end
        i = j;
    else
        i = i + 1;
    end
end

unique_runs = unique(runs(runs>0));
best_run_len = 0;
best_run_id = 0;
for rid = unique_runs
    len_run = sum(runs == rid);
    if len_run > best_run_len
        best_run_len = len_run;
        best_run_id = rid;
    end
end

if best_run_len >= min_consecutive_voiced
    sel = find(runs == best_run_id);
else
    [~, idx_energy] = sort(frame_energy_norm, 'descend');
    sel = sort(idx_energy(1:min(min_consecutive_voiced, length(idx_energy))));
    warning('Could not find %d contiguous voiced frames; using top-energy frames.', min_consecutive_voiced);
end

%% COMPUTE AVERAGES
if isempty(sel)
    error('No frames selected for averaging.');
end
F1_mean = mean(F(1, sel), 'omitnan');
F2_mean = mean(F(2, sel), 'omitnan');
F3_mean = mean(F(3, sel), 'omitnan');
valid_pitch_idx = sel(pitchHz(sel) > 0);
if ~isempty(valid_pitch_idx)
    pitch_mean = mean(pitchHz(valid_pitch_idx));
else
    pitch_mean = NaN;
end

fprintf('\nSelected frame indices: '); disp(sel);
fprintf('Average F1: %.1f Hz\n', F1_mean);
fprintf('Average F2: %.1f Hz\n', F2_mean);
fprintf('Average F3: %.1f Hz\n', F3_mean);
fprintf('Average Pitch: %.1f Hz\n', pitch_mean);

%% (1) Cepstrally smoothed log-spectrum
figure('Name','Cepstrally-smoothed log spectrum');
imagesc(frame_times, freq_axis, smoothed_log_spec);
axis xy;
colorbar;
ylim([0 max_formant_freq]);
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Cepstrally-smoothed log spectrum (frames)');

%% (2) Cepstral sequence (low quefrency)
max_quef_ms = 20;
max_quef_idx = min(nfft, round(max_quef_ms * 1e-3 * fs));
q_axis_ms = (0:max_quef_idx-1) / fs * 1000;
figure('Name','Cepstral sequence');
imagesc(frame_times, q_axis_ms, cepstra(1:max_quef_idx,:));
axis xy;
colorbar;
xlabel('Time (s)');
ylabel('Quefrency (ms)');
title('Cepstral sequence (low quefrency region)');

%% (3) Pitch contour & voicing
figure('Name','Pitch & Voicing');
subplot(2,1,1);
plot(frame_times, pitchHz, '-o', 'MarkerSize', 3);
hold on;
if ~isempty(sel)
    plot(frame_times(sel), pitchHz(sel), 'ro', 'MarkerFaceColor','r');
end
xlabel('Time (s)');
ylabel('Pitch (Hz)');
ymax = max(500, max(pitchHz)+20);
if isempty(ymax) || ~isfinite(ymax), ymax = 500; end
ylim([0 ymax]);
title('Framewise pitch (cepstral method)');

subplot(2,1,2);
plot(frame_times, cep_peak_mag, '-', 'DisplayName','Cepstral peak mag (norm)');
hold on;
plot(frame_times, frame_energy_norm, '--', 'DisplayName','Energy (norm)');
stem(frame_times, double(voiced_mask), 'k.', 'DisplayName','Voiced mask');
xlabel('Time (s)');
ylabel('Normalized');
legend;
title('Voicing decision metrics and mask');

%% (4) Formants over time
figure('Name','Formant tracks');
plot(frame_times, F(1,:), 'b.-', 'DisplayName','F1'); hold on;
plot(frame_times, F(2,:), 'g.-', 'DisplayName','F2');
plot(frame_times, F(3,:), 'r.-', 'DisplayName','F3');
if ~isempty(sel)
    plot(frame_times(sel), F(1,sel), 'bo', 'MarkerFaceColor','b');
end
xlabel('Time (s)');
ylabel('Frequency (Hz)');
ylim([0 max_formant_freq]);
legend('show');
grid on;
title('Estimated formant tracks from cepstrally-smoothed spectrum');
