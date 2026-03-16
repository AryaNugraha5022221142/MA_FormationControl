%% Multi-Agent UAV Formation Control with Obstacle Avoidance
clear all; close all; clc;
%% SIMULATION PARAMETERS
dt = 0.05;                      % Time step (s)
N = 6;                          % Number of UAVs
r_in = 0.5;                     % Minimum safe separation (m)
r_out = 3.0;                    % Collision avoidance radius (m)
R_comm = 20;                    % Communication range (m)
tau = 0.6;                      % Communication decay rate
% Control gains
k_p = 1.5;
k_v = 2.0;
k_att = 0.3;
k_rep = 5.0;
k_col = 2.0;
gamma = 0.8;
rho_0 = 6;
% Hexagon formation
formation_radius = 5;
angles = (0:N-1) * (2*pi/N);
d_formation = zeros(2, N);
for i = 1:N
    d_formation(:,i) = formation_radius * [cos(angles(i)); sin(angles(i))];
end
% Simulation durations
T_simulations = [60, 120, 180];
n_sims = length(T_simulations);
%% STORAGE FOR RESULTS
results = struct();
%% MAIN SIMULATION LOOP
for sim_idx = 1:n_sims
    fprintf('Running simulation for T = %d seconds...\n', T_simulations(sim_idx));
    
    T_final = T_simulations(sim_idx);
    time = 0:dt:T_final;
    N_steps = length(time);
    
    % OBSTACLE CONFIGURATION
    if T_final == 60
        obs_centers = [15, 5; 25, -3; 35, 8];
        obs_radii = [3; 2.5; 3];
        n_obs = 3;
    elseif T_final == 120
        obs_centers = [15, 5; 25, -3; 35, 8; 50, -5; 65, 7];
        obs_radii = [3; 2.5; 3; 2.8; 3.2];
        n_obs = 5;
    else  % 180s
        obs_centers = [15, 5; 25, -3; 35, 8; 50, -5; 65, 7; 80, -4; 95, 6];
        obs_radii = [3; 2.5; 3; 2.8; 3.2; 2.7; 3.1];
        n_obs = 7;
    end
    
    % Initial conditions
    p = randn(2, N) * 3;
    v = zeros(2, N);
    
    % Data storage
    p_history = zeros(2, N, N_steps);
    v_history = zeros(2, N, N_steps);
    formation_error = zeros(1, N_steps);
    min_distance = zeros(1, N_steps);
    control_effort = zeros(1, N_steps);
    
    goal_velocity = 0.5;
    goal_initial = [0; 0];
    
    %% SIMULATION LOOP
    for k = 1:N_steps
        t = time(k);
        
        % Update goal position
        p_goal = goal_initial + [goal_velocity * t; 0];
        
        % Compute weighted adjacency matrix
        A = compute_adjacency(p, R_comm, tau);
        D = diag(sum(A, 2));
        L = D - A;
        
        % Initialize control inputs
        u = zeros(2, N);
        
        %% CONTROL LAW for each agent
        for i = 1:N
            % 1. CONSENSUS FORMATION CONTROL
            u_form = zeros(2, 1);
            neighbors = find(A(i, :) > 0);
            
            if ~isempty(neighbors)
                for j = neighbors
                    desired_rel = d_formation(:,i) - d_formation(:,j);
                    actual_rel = p(:,i) - p(:,j);
                    delta_p = actual_rel - desired_rel;
                    delta_v = v(:,i) - v(:,j);
                    u_form = u_form - k_p * delta_p - k_v * delta_v;
                end
            end
            
            % 2. ATTRACTIVE POTENTIAL (toward goal)
            F_att = -k_att * (p(:,i) - p_goal);
            
            % 3. REPULSIVE POTENTIAL (obstacles)
            F_rep_obs = zeros(2, 1);
            for obs_idx = 1:n_obs
                p_obs = obs_centers(obs_idx, :)';
                r_obs = obs_radii(obs_idx);
                
                dist_to_center = norm(p(:,i) - p_obs);
                rho = dist_to_center - r_obs;
                
                if rho > 0 && rho <= rho_0
                    F_mag = k_rep * (1/rho - 1/rho_0) * (1/rho^2);
                    direction = (p(:,i) - p_obs) / dist_to_center;
                    F_rep_obs = F_rep_obs + F_mag * direction;
                end
            end
            
            % 4. INTER-AGENT COLLISION AVOIDANCE
            F_col = zeros(2, 1);
            for j = 1:N
                if j ~= i
                    dist_ij = norm(p(:,i) - p(:,j));
                    if dist_ij < r_out && dist_ij > r_in
                        g1 = 0.5 * (1 + cos(pi * (dist_ij - r_in) / (r_out - r_in)));
                        F_mag = k_col * g1 / (dist_ij + 1e-6);
                        if dist_ij > 0
                            direction = (p(:,i) - p(:,j)) / dist_ij;
                            F_col = F_col + F_mag * direction;
                        end
                    end
                end
            end
            
            % COMBINED CONTROL INPUT
            u(:,i) = u_form + gamma * (F_att + F_rep_obs) + F_col;
        end
        
        %% UPDATE DYNAMICS
        for i = 1:N
            v(:,i) = v(:,i) + u(:,i) * dt;
            p(:,i) = p(:,i) + v(:,i) * dt;
        end
        
        %% STORE DATA
        p_history(:,:,k) = p;
        v_history(:,:,k) = v;
        
        %% COMPUTE METRICS
        e_sum = 0;
        count = 0;
        for i = 1:N
            neighbors = find(A(i, :) > 0);
            for j = neighbors
                desired_rel = d_formation(:,i) - d_formation(:,j);
                actual_rel = p(:,i) - p(:,j);
                e_sum = e_sum + norm(actual_rel - desired_rel);
                count = count + 1;
            end
        end
        formation_error(k) = e_sum / max(count, 1);
        
        min_dist = inf;
        for i = 1:N
            for j = i+1:N
                dist = norm(p(:,i) - p(:,j));
                if dist < min_dist
                    min_dist = dist;
                end
            end
        end
        min_distance(k) = min_dist;
        
        control_effort(k) = sum(vecnorm(u, 2, 1));
    end
    
    %% STORE RESULTS
    results(sim_idx).T = T_final;
    results(sim_idx).time = time;
    results(sim_idx).p_history = p_history;
    results(sim_idx).v_history = v_history;
    results(sim_idx).formation_error = formation_error;
    results(sim_idx).min_distance = min_distance;
    results(sim_idx).control_effort = control_effort;
    results(sim_idx).obs_centers = obs_centers;
    results(sim_idx).obs_radii = obs_radii;
    results(sim_idx).n_obs = n_obs;
    
    fprintf('Simulation complete for T = %d seconds\n\n', T_final);
end

%% FIGURE 1: TRAJECTORIES
figure('Name', 'Trajectories', 'NumberTitle', 'off', 'Position', [100, 100, 1600, 450]);
for sim_idx = 1:3
    subplot(1, 3, sim_idx);
    hold on; grid on; axis equal;
    xlabel('X Position (m)', 'FontSize', 11);
    ylabel('Y Position (m)', 'FontSize', 11);
    title(sprintf('Trajectories (T = %d s)', results(sim_idx).T), 'FontSize', 12, 'FontWeight', 'bold');
    
    obs_centers_temp = results(sim_idx).obs_centers;
    obs_radii_temp = results(sim_idx).obs_radii;
    for obs_idx = 1:size(obs_centers_temp, 1)
        theta = linspace(0, 2*pi, 100);
        x_circle = obs_centers_temp(obs_idx, 1) + obs_radii_temp(obs_idx) * cos(theta);
        y_circle = obs_centers_temp(obs_idx, 2) + obs_radii_temp(obs_idx) * sin(theta);
        fill(x_circle, y_circle, [0.8 0.8 0.8], 'EdgeColor', 'k', 'LineWidth', 1.5);
    end
    
    colors = lines(N);
    p_hist = results(sim_idx).p_history;
    for agent_idx = 1:N
        plot(squeeze(p_hist(1, agent_idx, :)), squeeze(p_hist(2, agent_idx, :)), ...
            'Color', colors(agent_idx, :), 'LineWidth', 1.5);
        plot(p_hist(1, agent_idx, 1), p_hist(2, agent_idx, 1), 'o', ...
            'MarkerFaceColor', colors(agent_idx, :), 'MarkerSize', 8);
        plot(p_hist(1, agent_idx, end), p_hist(2, agent_idx, end), 's', ...
            'MarkerFaceColor', colors(agent_idx, :), 'MarkerSize', 8);
    end
    
    set(gca, 'FontSize', 10);
end
sgtitle('Multi-UAV Trajectories', 'FontSize', 14, 'FontWeight', 'bold');

%% FIGURE 2: FORMATION ERROR
figure('Name', 'Formation Error', 'NumberTitle', 'off', 'Position', [100, 600, 1600, 450]);
for sim_idx = 1:3
    subplot(1, 3, sim_idx);
    time_temp = results(sim_idx).time;
    error_temp = results(sim_idx).formation_error;
    
    plot(time_temp, error_temp, 'b-', 'LineWidth', 2);
    xlabel('Time (s)', 'FontSize', 11);
    ylabel('Formation Error (m)', 'FontSize', 11);
    title(sprintf('Formation Error (T = %d s)', results(sim_idx).T), 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    set(gca, 'FontSize', 10);
end
sgtitle('Formation Error Evolution', 'FontSize', 14, 'FontWeight', 'bold');

%% FIGURE 3: MINIMUM INTER-AGENT DISTANCE 
figure('Name', 'Safety Metrics', 'NumberTitle', 'off', 'Position', [100, 1100, 1600, 450]);
for sim_idx = 1:3
    subplot(1, 3, sim_idx);
    time_temp = results(sim_idx).time;
    min_dist_temp = results(sim_idx).min_distance;
    
    plot(time_temp, min_dist_temp, 'r-', 'LineWidth', 2);
    hold on;
    plot(time_temp, r_in * ones(size(time_temp)), 'k--', 'LineWidth', 1.5, 'DisplayName', 'Safety Threshold');
    xlabel('Time (s)', 'FontSize', 11);
    ylabel('Minimum Distance (m)', 'FontSize', 11);
    title(sprintf('Minimum Inter-Agent Distance (T = %d s)', results(sim_idx).T), 'FontSize', 12, 'FontWeight', 'bold');
    legend('Min Distance', 'Safety Threshold', 'FontSize', 9);
    grid on;
    set(gca, 'FontSize', 10);
end
sgtitle('Multi-UAV Collision Avoidance', 'FontSize', 14, 'FontWeight', 'bold');

%% FIGURE 4: CONTROL EFFORT
figure('Name', 'Control Effort', 'NumberTitle', 'off', 'Position', [100, 1600, 1600, 450]);
for sim_idx = 1:3
    subplot(1, 3, sim_idx);
    time_temp = results(sim_idx).time;
    effort_temp = results(sim_idx).control_effort;
    
    plot(time_temp, effort_temp, 'g-', 'LineWidth', 2);
    xlabel('Time (s)', 'FontSize', 11);
    ylabel('Control Effort (N)', 'FontSize', 11);
    title(sprintf('Control Effort (T = %d s)', results(sim_idx).T), 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    set(gca, 'FontSize', 10);
end
sgtitle('Control Effort Distribution', 'FontSize', 14, 'FontWeight', 'bold');

%% ANIMATIONS
fprintf('\n========================================\n');
fprintf('Generating animations...\n');
fprintf('========================================\n');

% Color map for agents
colors = lines(N);

for sim_idx = 1:n_sims
    fprintf('\n--- Animation for T = %d seconds ---\n', results(sim_idx).T);
    
    % Create new figure for animation
    anim_fig = figure('Name', sprintf('Animation T=%ds', results(sim_idx).T), ...
           'NumberTitle', 'off', 'Position', [200, 200, 1200, 900]);
    
    p_hist = results(sim_idx).p_history;
    v_hist = results(sim_idx).v_history;
    time_sim = results(sim_idx).time;
    obs_centers = results(sim_idx).obs_centers;
    obs_radii = results(sim_idx).obs_radii;
    n_obs_sim = results(sim_idx).n_obs;
    
    % Animation settings
    frame_skip = 5;  % Skip frames untuk kecepatan
    total_frames = floor(length(time_sim) / frame_skip);
    
    % Animation loop
    for frame = 1:total_frames
        k = frame * frame_skip;
        if k > length(time_sim)
            break;
        end
        
        % Clear figure
        clf(anim_fig);
        
        % Main plot
        hold on; 
        grid on; 
        axis equal;
        set(gca, 'Color', [0.95 0.95 0.95]);  % Light gray background
        
        % Plot obstacles dengan style lebih bagus
        for obs_idx = 1:n_obs_sim
            theta = linspace(0, 2*pi, 100);
            x_circle = obs_centers(obs_idx, 1) + obs_radii(obs_idx) * cos(theta);
            y_circle = obs_centers(obs_idx, 2) + obs_radii(obs_idx) * sin(theta);
            
            % Obstacle dengan gradient
            fill(x_circle, y_circle, [0.7 0.7 0.7], ...
                'EdgeColor', 'k', 'LineWidth', 2.5, 'FaceAlpha', 0.7);
            
            % Obstacle label
            text(obs_centers(obs_idx, 1), obs_centers(obs_idx, 2), ...
                'OBS', 'HorizontalAlignment', 'center', ...
                'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
        end
        
        % Compute adjacency for communication links
        A_current = compute_adjacency(p_hist(:,:,k), R_comm, tau);
        
        % Plot communication links DULU (biar di belakang)
        for i = 1:N
            for j = i+1:N
                if A_current(i,j) > 0.1
                    plot([p_hist(1,i,k), p_hist(1,j,k)], ...
                         [p_hist(2,i,k), p_hist(2,j,k)], ...
                         'k--', 'LineWidth', 1.5, 'Color', [0.3 0.3 0.3 0.5]);
                end
            end
        end
        
        % Plot UAVs dengan velocity vectors
        for i = 1:N
            % UAV body (lingkaran besar)
            plot(p_hist(1,i,k), p_hist(2,i,k), 'o', ...
                'MarkerFaceColor', colors(i,:), ...
                'MarkerEdgeColor', 'k', ...
                'MarkerSize', 18, ...
                'LineWidth', 2);
            
            % Velocity vector (panah)
            vel_scale = 1.5;  % Scale untuk visibility
            if norm([v_hist(1,i,k), v_hist(2,i,k)]) > 0.01
                quiver(p_hist(1,i,k), p_hist(2,i,k), ...
                    v_hist(1,i,k)*vel_scale, v_hist(2,i,k)*vel_scale, ...
                    0, 'Color', colors(i,:), 'LineWidth', 3, ...
                    'MaxHeadSize', 1.5, 'AutoScale', 'off');
            end
            
            % UAV label
            text(p_hist(1,i,k), p_hist(2,i,k)-2, ...
                sprintf('UAV%d', i), ...
                'HorizontalAlignment', 'center', ...
                'FontSize', 11, 'FontWeight', 'bold', ...
                'Color', colors(i,:), ...
                'BackgroundColor', 'w', ...
                'EdgeColor', colors(i,:), ...
                'Margin', 2);
        end
        
        % Set axis limits dengan margin dinamis
        all_x = p_hist(1,:,k);
        all_y = p_hist(2,:,k);
        margin = 15;
        xlim([min(all_x) - margin, max(all_x) + margin]);
        ylim([min(all_y) - margin, max(all_y) + margin]);
        
        % Labels dan title
        xlabel('X Position (m)', 'FontSize', 13, 'FontWeight', 'bold');
        ylabel('Y Position (m)', 'FontSize', 13, 'FontWeight', 'bold');
        title(sprintf('Multi-UAV Formation Control | T = %d s | t = %.1f s (%.0f%%)', ...
            results(sim_idx).T, time_sim(k), 100*k/length(time_sim)), ...
            'FontSize', 14, 'FontWeight', 'bold');
        
        % Add info box
        info_str = sprintf('Formation Error: %.2f m\nMin Distance: %.2f m\nActive Links: %d', ...
            results(sim_idx).formation_error(k), ...
            results(sim_idx).min_distance(k), ...
            sum(sum(A_current > 0.1))/2);
        
        annotation('textbox', [0.02 0.02 0.2 0.15], 'String', info_str, ...
            'FitBoxToText', 'on', 'BackgroundColor', 'white', ...
            'EdgeColor', 'k', 'LineWidth', 1.5, ...
            'FontSize', 10, 'FontWeight', 'bold');
        
        set(gca, 'FontSize', 11, 'LineWidth', 1.5);
        box on;
        
        % PENTING: drawnow untuk update figure
        drawnow;
        
        % Pause yang lebih lama supaya keliatan geraknya
        pause(0.05);  % 50ms per frame
        
        % Display progress
        if mod(frame, 50) == 0
            fprintf('Progress: %.0f%%\n', 100*frame/total_frames);
        end
    end
    
    fprintf('Animation complete for T = %d seconds!\n', results(sim_idx).T);
end

%% COMPARATIVE STATISTICS
fprintf('\n========================================\n');
fprintf('COMPARATIVE PERFORMANCE METRICS\n');
fprintf('========================================\n\n');
fprintf('%-25s %15s %15s %15s\n', 'Metric', 'T=60s', 'T=120s', 'T=180s');
fprintf('%-25s %15s %15s %15s\n', repmat('-', 1, 25), repmat('-', 1, 15), ...
        repmat('-', 1, 15), repmat('-', 1, 15));

for i = 1:3
    steady_idx = find(results(i).time > 20);
    avg_error(i) = mean(results(i).formation_error(steady_idx));
    max_error(i) = max(results(i).formation_error);
    violations(i) = sum(results(i).min_distance < r_in);
    min_dist_obs(i) = min(results(i).min_distance);
    total_effort(i) = sum(results(i).control_effort) * dt;
    norm_effort(i) = total_effort(i) / results(i).T;
end

fprintf('%-25s %15.3f %15.3f %15.3f\n', 'Avg Error (m)', avg_error(1), avg_error(2), avg_error(3));
fprintf('%-25s %15.3f %15.3f %15.3f\n', 'Max Error (m)', max_error(1), max_error(2), max_error(3));
fprintf('%-25s %15d %15d %15d\n', 'Violations', violations(1), violations(2), violations(3));
fprintf('%-25s %15.3f %15.3f %15.3f\n', 'Min Distance (m)', min_dist_obs(1), min_dist_obs(2), min_dist_obs(3));
fprintf('%-25s %15.1f %15.1f %15.1f\n', 'Total Control Effort', total_effort(1), total_effort(2), total_effort(3));
fprintf('%-25s %15.3f %15.3f %15.3f\n', 'Effort per Second', norm_effort(1), norm_effort(2), norm_effort(3));
fprintf('%-25s %15d %15d %15d\n', 'Num Obstacles', results(1).n_obs, results(2).n_obs, results(3).n_obs);

fprintf('\n========================================\n');
fprintf('All simulations and visualizations complete!\n');
fprintf('========================================\n');

%% ============== HELPER FUNCTION ==============
function A = compute_adjacency(p, R_comm, tau)
    N = size(p, 2);
    A = zeros(N, N);
    
    for i = 1:N
        for j = i+1:N
            dist = norm(p(:,i) - p(:,j));
            
            if dist < tau * R_comm
                A(i,j) = 1;
                A(j,i) = 1;
            elseif dist >= tau * R_comm && dist < R_comm
                weight = 0.5 * (1 + cos(pi * (dist - tau*R_comm) / ((1-tau)*R_comm)));
                A(i,j) = weight;
                A(j,i) = weight;
            end
        end
    end
end
