function [t, back, freq, back_f] = FluidSphere(theta, r_rev, r_src, a, rho1, c1, rho0, c0, source_signal, t_step)
% FLUIDSPHERE Provides theoretical scattered signal from a fluid spherical scatterer.
%
% This function calculates the scattered acoustic signal from a spherical
% scatterer in a fluid medium based on the theory presented by Victor C. Anderson in 
% "Sound scattering from a fluid sphere," The Journal of the Acoustical Society of America,
% Vol. 22, No. 4, pp. 426-431, 1950.

% [t, back, freq, back_f] = FluidSphere(theta, r_rev, r_src, a, rho1, c1, rho0, c0, source_signal, t_step, t_end)
% Input parameters:
%
%   theta         - Scattering angle in radians; theta=0 implies backscattering.
%   r_rev         - Distance from the receiver to the center of the scatterer (m).
%   r_src         - Distance from the planar source to the center of the scatterer (m).
%   a             - Radius of the spherical scatterer (m).
%   rho1          - Density of the scatterer (kg/m^3).
%   c1            - Compressional wave speed of the scatterer (m/s).
%   rho0          - Density of the background medium (kg/m^3).
%   c0            - Compressional wave speed of the background medium (m/s).
%   source_signal - The incident source signal, expected to be a 1xNt array.
%   t_step        - Time step of the source signal (s).
%
% Output parameters:
%
%   t             - Time vector corresponding to the source signal.
%   back          - Time-domain scattered signal, matching the length of the source signal.
%   freq          - Frequency vector for the frequency-domain analysis.
%   back_f        - Frequency-domain scattered signal, returned as a 1xNf complex array.
%
% Example:
%   [t, back, freq, back_f] = FluidSphere(0, 4e-3, 3e-3, 1e-3, 1100, 1600, 1000, 1500, signal, 2e-9, 1e-5);

% Author: Zixuan Tian (zixuant5@illinois.edu)
% Date: June 26, 2023
% Revision: April 28, 2024

    N = 25; % number of terms

    sampling_freq = 1/t_step;
    Nt = length(source_signal);
    t = t_step*(0:1:Nt-1);

    % Delayed version
    delay_t = round((r_src)/c0/t_step); % in time steps
    tone_burst = circshift(source_signal, delay_t);
    
    freq = (0:Nt-1)*sampling_freq/Nt;

    mu = cos(theta);
    P = zeros(N+1,1);
    for i=1:N+1
        tmp_P = legendre(i-1,mu);
        P(i,1) = tmp_P(1);
    end
    
    if size(freq,1) == 1    freq = freq'; end       % convert to column vector.
    
    g = rho1/rho0;        h = c1/c0;
    k = 2*pi*freq/c0;       ka = k*a;       kpa = 2*pi*freq/c1*a;   % column vector.
    
    % All the N terms are calculated simultaneously in matrix form.
    
    ja = sbessel(-1:N+1, ka);    jpa = sbessel(-1:N+1, kpa);    na = sbessely(-1:N+1, ka);
    ja1 = ja(:,1:end-2).*(ones(size(ja,1),1)*(0:N));
    ja2 = ja(:,3:end).*(ones(size(ja,1),1)*(1:N+1));
    jpa1 = jpa(:,1:end-2).*(ones(size(jpa,1),1)*(0:N));
    jpa2 = jpa(:,3:end).*(ones(size(jpa,1),1)*(1:N+1));
    na1 = na(:,1:end-2).*(ones(size(na,1),1)*(0:N));
    na2 = na(:,3:end).*(ones(size(na,1),1)*(1:N+1));
    
    beta = na1 - na2;   alpha = ja1 - ja2;  alphap = jpa1 - jpa2; % same notation as the Anderson paper
    
    tmp = alphap./alpha./jpa(:,2:end-1);
    num = tmp.*na(:,2:end-1) - beta./alpha*g*h;
    den = tmp.*ja(:,2:end-1) - g*h;
    C = num./den;
    
    kr = k*r_rev;
    jr = sbessel(-1:N+1, kr);  nr = sbessely(-1:N+1, kr);
    A2 = (ones(size(C,1),1)*(((-1j).^(0:N)).*(2*(0:N)+1)))./(1+1i*C).*(jr(:,2:end-1)+1j*nr(:,2:end-1));
    A2(isnan(A2)) = 0;        % replace NaNs with zeros
    
    A2 = -1*A2*P;
    
    inc_f = fft(tone_burst);
    back_f = inc_f.*A2';
    back = real(ifft(back_f));

end