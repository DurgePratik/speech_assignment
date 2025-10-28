
clear; clc; close all;

%% === Step 1: Load audio file ===
[file, path] = uigetfile('hindi_ba.wav', 'Select your speech file');
if isequal(file,0)
    disp('User canceled.');
    return;
end
[x, fs] = audioread(fullfile(path, file));
x = x(:,1);                    % mono
x = x - mean(x);               % remove DC

fprintf('Loaded %s (%.2f sec, %d Hz)\n', file, length(x)/fs, fs);

%% === Step 2: Analysis parameters ===
frame_dur = 0.03;     % 30 ms
hop_dur   = 0.01;     % 10 ms
frame_len = round(frame_dur * fs);
hop_len   = round(hop_dur * fs);

fmin = 60;  fmax = 400;
tau_min = floor(fs / fmax);
tau_max = ceil(fs / fmin);

%% === Step 3: Frame-wise AMD pitch estimation ===
num_frames = floor((length(x)-frame_len)/hop_len) + 1;
f0_track = zeros(num_frames,1);
t_axis = (0:num_frames-1)*hop_dur;

for i = 1:num_frames
    start = (i-1)*hop_len + 1;
    frame = x(start:start+frame_len-1);
    frame = frame .* hamming(frame_len);
    f0_track(i) = amd_pitch_refined(frame, fs, tau_min, tau_max);
end

%% === Step 4: Post-process ===
f0_track(f0_track < fmin | f0_track > fmax) = NaN;
f0_smooth = medfilt1(f0_track, 3, 'omitnan');

%% === Step 5: Compute final pitch value (median of voiced region) ===
% Automatically pick region with speech energy > 0.1 * max energy
frame_energy = buffer(x.^2, frame_len, frame_len-hop_len, 'nodelay');
frame_energy = mean(frame_energy, 1);
voiced_idx = frame_energy > 0.1 * max(frame_energy);

voiced_f0 = f0_smooth(voiced_idx);
final_pitch = median(voiced_f0(~isnan(voiced_f0)));

if isnan(final_pitch)
    disp('No voiced frames detected.');
    final_pitch = 0;
end
fprintf('Estimated pitch ≈ %.1f Hz\n', final_pitch);

%% === Step 6: Plot waveform + pitch contour ===
t = (0:length(x)-1)/fs;
figure('Name','Pitch Estimation','NumberTitle','off');

subplot(2,1,1);
plot(t, x);
xlabel('Time (s)'); ylabel('Amplitude');
title('Speech Waveform');
grid on;

subplot(2,1,2);
plot(t_axis, f0_smooth, 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('Pitch (Hz)');
title(sprintf('Estimated Pitch Contour (Pitch ≈ %.1f Hz)', final_pitch));
ylim([0 400]); grid on;

%% === Step 7: Plot AMD curve for middle voiced frame ===
mid_frame = find(voiced_idx, 1, 'first') + round(sum(voiced_idx)/2);
if isempty(mid_frame) || mid_frame > num_frames
    mid_frame = round(num_frames/2);
end

start_idx = (mid_frame-1)*hop_len + 1;
frame = x(start_idx:start_idx+frame_len-1);
frame = frame .* hamming(frame_len);

[~, amd_curve, taus] = amd_curve_func(frame, fs, tau_min, tau_max);
[~, minIdx] = min(amd_curve);
f0_frame = fs / taus(minIdx);

figure('Name','AMD Curve','NumberTitle','off');
plot(taus/fs*1000, amd_curve, 'b', 'LineWidth', 1.5); hold on;
plot(taus(minIdx)/fs*1000, amd_curve(minIdx), 'ro', 'MarkerFaceColor','r');
xlabel('Lag (ms)'); ylabel('AMD Value');
title(sprintf('AMD Curve  (Final Pitch = %.1f Hz)', ...
 final_pitch));
grid on;

% %%% === Step 8: Save pitch results ===
% T = table(t_axis.', f0_smooth, 'VariableNames', {'Time_s','Pitch_Hz'});
% writetable(T, 'amd_pitch_output.csv');
% fprintf('Saved results to amd_pitch_output.csv\n');


function f0 = amd_pitch_refined(frame, fs, tau_min, tau_max)
    frame = frame - mean(frame);
    N = length(frame);
    taus = tau_min:tau_max;
    amd_vals = zeros(size(taus));

    for k = 1:length(taus)
        tau = taus(k);
        amd_vals(k) = mean(abs(frame(1:N-tau) - frame(1+tau:N)));
    end

    amd_vals = amd_vals / max(amd_vals);
    amd_vals = smoothdata(amd_vals, 'movmean', 5);

    [~, locs] = findpeaks(-amd_vals);
    if isempty(locs), f0 = NaN; return; end
    valid_locs = locs(taus(locs) > tau_min + 10);
    if isempty(valid_locs), f0 = NaN; return; end

    [~, idxMin] = min(amd_vals(valid_locs));
    lag = taus(valid_locs(idxMin));

    lag2 = 2 * lag;
    if lag2 <= tau_max
        val1 = amd_vals(lag - tau_min + 1);
        val2 = amd_vals(lag2 - tau_min + 1);
        if val2 < 1.2 * val1
            lag = lag2; % octave correction
        end
    end

    f0 = fs / lag;
end

function [f0, amd_curve, taus] = amd_curve_func(frame, fs, tau_min, tau_max)
    frame = frame - mean(frame);
    N = length(frame);
    taus = tau_min:tau_max;
    amd_curve = zeros(size(taus));
    for k = 1:length(taus)
        tau = taus(k);
        amd_curve(k) = mean(abs(frame(1:N-tau) - frame(1+tau:N)));
    end
    [~, idx] = min(amd_curve);
    f0 = fs / taus(idx);
end
