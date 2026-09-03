function P = get_event_disruption_params(scenario, n1, n2, n3)

    switch scenario

        case 1
            % Scenario 1: moderate continuous event
            % e.g. ordinary accident / local congestion / cloudy period / moderate drawdown
            P.num_events    = 8;
            P.L             = max(6, round(0.08 * n1));

            P.day_num       = min(max(1, round(0.12 * n2)), 10);
            P.series_num    = min(max(1, round(0.18 * n3)), 25);

            % attenuation + downward shift
            P.rho           = 0.18;%0.45;
            P.delta         = 0.04;%0.08;

            % local heterogeneity
            P.rho_jitter    = 0.03;%0.05;
            P.delta_jitter  = 0.01;%0.02;

            % onset -> plateau -> recovery
            P.onset_ratio   = 0.20;
            P.plateau_ratio = 0.75;
            P.onset_level   = 0.20;
            P.recovery_level= 0.30;%0.35;

        case 2
            % Scenario 2: severe continuous event
            % e.g. severe accident / bottleneck / storm / severe drawdown
            P.num_events    = 4;
            P.L             = max(10, round(0.12 * n1));%max(10, round(0.15 * n1));

            P.day_num       =min(max(1, round(0.10 * n2)), 15);
            P.series_num    = min(max(1, round(0.22 * n3)), 40);%min(max(1, round(0.25 * n3)), 45);


            P.rho           = 0.32;%0.75;
            P.delta         = 0.07;%0.18;

            P.rho_jitter    = 0.04;%0.06;
            P.delta_jitter  = 0.015;%0.03;

            P.onset_ratio   = 0.15;
            P.plateau_ratio = 0.75%0.80;
            P.onset_level   = 0.25;%0.30;
            P.recovery_level= 0.40;%0.60;

        otherwise
            error('Unknown scenario. Use 1 or 2.');
    end

    P.L              = min(P.L, max(1, n1 - 1));
    P.day_num        = min(max(1, P.day_num), n2);
    P.series_num     = min(max(1, P.series_num), n3);

    P.rho            = max(P.rho, 0);
    P.delta          = max(P.delta, 0);

    P.rho_jitter     = max(P.rho_jitter, 0);
    P.delta_jitter   = max(P.delta_jitter, 0);

    P.onset_ratio    = min(max(P.onset_ratio, 0.05), 0.5);
    P.plateau_ratio  = min(max(P.plateau_ratio, P.onset_ratio + 0.05), 0.95);
    P.onset_level    = min(max(P.onset_level, 0), 1);
    P.recovery_level = min(max(P.recovery_level, 0), 1);
end