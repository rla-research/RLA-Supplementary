function anomal = event_disruption_anomaly(X0, scenario, seed)

    rng(seed, 'twister');

    [n1, n2, n3] = size(X0);
    anomal = zeros(size(X0));

    P = get_event_disruption_params(scenario, n1, n2, n3);
    is_nonnegative_data = (min(X0(:)) >= 0);

    for m = 1:P.num_events

        % time block
        L = max(1, min(P.L, n1));
        t0 = randi([1, n1 - L + 1]);
        t_idx = t0:(t0 + L - 1);

        % affected mode-2 subset
        d_num = min(P.day_num, n2);
        d_sel = randperm(n2, d_num);

        % affected mode-3 subset
        s_num = min(P.series_num, n3);
        s_sel = randperm(n3, s_num);

        % event profile
        w = local_event_profile_clean(L, P);

        for d = d_sel
            for s = s_sel
                x = X0(:, d, s);
                x_seg = x(t_idx);

                    scale=std(abs(x_seg));%nasdaq
                    %scale = median(abs(x_seg));
                    scale = max(scale, eps);
               

                % local heterogeneity
                rho_ds   = min(max(P.rho   + P.rho_jitter   * randn, 0), 1.2);
                delta_ds = max(P.delta + P.delta_jitter * randn, 0);

                % event-driven attenuation + downward shift
                y_seg = (1 - rho_ds * w) .* x_seg - delta_ds * scale .* w;

                if is_nonnegative_data
                    y_seg = max(0, y_seg);
                
                end
                anomal(t_idx, d, s) = anomal(t_idx, d, s) + (y_seg - x_seg);
            end
        end
    end
end


function w = local_event_profile_clean(L, P)
    k1 = max(1, round(P.onset_ratio * L));
    k2 = max(k1 + 1, round(P.plateau_ratio * L));
    k2 = min(k2, L);

    w = zeros(L,1);

    w(1:k1) = linspace(P.onset_level, 1.0, k1);

    if k2 > k1
        w(k1+1:k2) = 1.0;
    end

    if k2 < L
        w(k2+1:end) = linspace(1.0, P.recovery_level, L-k2);
    end
end