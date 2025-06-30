function [t, back, freq, back_f] = SolidCylinder(theta, r_rev, r_src, a, rho1, c1, c2, rho0, c0, source_signal, t_step, t_end)
% SOLIDCYLINDER Provides theoretical scattered signal from a solid cylindrical scatterer.
%
% This function calculates the scattered acoustic signal from an infinitely
% long solid cylindrical scatterer in a fluid medium. The theorey is based on
% Faran Jr, James J. "Sound scattering by solid cylinders and spheres." The
% Journal of the acoustical society of America 23.4 (1951): 405-418.

% [t, back, freq, back_f] = SolidCylinder(theta, r_rev, r_src, a, rho1, c1, c2, rho0, c0, source_signal, t_step, t_end)
% Input parameters:
%
%   theta         - Scattering angle in radians; theta=0 implies backscattering.
%   r_rev         - Distance from the receiver to the center of the scatterer (m).
%   r_src         - Distance from the planar source to the center of the scatterer (m).
%   a             - Radius of the spherical scatterer (m).
%   rho1          - Density of the scatterer (kg/m^3).
%   c1            - Compressional wave speed of the scatterer (m/s).
%   c2            - Shear wave speed of the scatterer (m/s).
%   rho0          - Density of the background medium (kg/m^3).
%   c0            - Compressional wave speed of the background medium (m/s).
%   source_signal - The incident source signal, expected to be a 1xNt array.
%   t_step        - Time step of the source signal (s).
%   t_end         - End time of the source signal (s).
%
% Output parameters:
%
%   t             - Time vector corresponding to the source signal.
%   back          - Time-domain scattered signal, matching the length of the source signal.
%   freq          - Frequency vector for the frequency-domain analysis.
%   back_f        - Frequency-domain scattered signal, returned as a 1xNf complex array.
%
% Example:
%   [t, back, freq, back_f] = SolidCylinder(0, 4e-3, 3e-3, 1e-3, 1960, 4030, 1645, 1000, 1500, signal, 6e-10, 2e-5);

% Author: Zixuan Tian (zixuant5@illinois.edu)
% Date: Sep 13, 2023
% Revision: April 28, 2024

sampling_freq = 1/t_step;
Nt = ceil(t_end/t_step);
t = t_step*(0:1:Nt-1);

% Delayed version
delay_t = round((r_src)/c0/t_step); % in time steps
tone_burst = circshift(source_signal, delay_t);

Nf = Nt*2^5; % We need high frequency resolution
freq = (0:Nf-1)*sampling_freq/Nf;

% speed of sound and density in the fluid surrounding the scatterer
c3 = c0;
density1 = rho1;
density3 = rho0;
ka = a*2*pi*freq/c3;

theta = pi - theta;

%Number of iterations used for the computation
order = 25;

im = sqrt(-1);

% x1 = k1*a, x2 = k2*a, x3 = k3*a where k = 2*pi*f/c. These variables are
% defined the same way in the paper.
x1 = ka.*c3/c1;
x2 = ka.*c3/c2;
x3 = ka;

% Initialize memory for some variables
eta = zeros(length(ka),order+1);
sum = zeros(length(ka),1);

for i = 0:order
   
   % the variable names are consistent with the paper - i.e. Jx1 is j(x1),
   % Nx3 is n(x3) (bessel and neumann functions)
   Jx1 = besselj(i,x1);
   Jx2 = besselj(i,x2);
   Jx3 = besselj(i,x3);
   Nx3 = bessely(i,x3);
   
   % the derivative of a bessel function can be expressed in
   % terms of bessel functions of different order.
   x1Jx1_prime = 0.5*x1.*(besselj(i-1,x1)-besselj(i+1,x1));
   x3Nx3_prime = 0.5*x3.*(bessely(i-1,x3)-bessely(i+1,x3));
   x2Jx2_prime = 0.5*x2.*(besselj(i-1,x2)-besselj(i+1,x2));
   x3Jx3_prime = 0.5*x3.*(besselj(i-1,x3)-besselj(i+1,x3));
   
   % The intermediate angles are defined differently than in the paper;
   % they are the equal to the tangent of the angles as defined in the
   % paper. For example, alpha(here) = tan[alpha(paper)]
   deltax3 = -Jx3./Nx3;
   alphax3 = -x3Jx3_prime./Jx3;
   betax3 = -x3Nx3_prime./Nx3;
   alphax1 = -x1Jx1_prime./Jx1;
   alphax2 = -x2Jx2_prime./Jx2;
 
   % due to the complexity of the equation, the numerator and denominator
   % of equation 28 are computed individually
   Num = alphax1./(alphax1+1) - (i^2)./(alphax2+i^2-0.5*x2.^2);
   Denom = (alphax1+i^2-0.5*x2.^2)./(alphax1+1) - ...
            ((i^2)*(alphax2+1))./(alphax2+i^2-0.5*x2.^2);
   % finally, the tan of zeta (as described by eq. 25) is computed
   tan_zeta = (-(x2.^2)./2).*Num./Denom;   

   % As with the intermediate angles, phi is actually tan[phi] by the
   % notation in the paper.
   phi = -(density3/density1)*tan_zeta;

   % eta is the same as eta in the paper.  
   eta(:,i+1)= atan(deltax3'.*(alphax3'+phi')./(phi'+betax3'));
end

k3r = x3/a*r_rev;
% for each value in the series, the sum as described by equation 26 in the
% paper is computed.
sum_terms = zeros(length(sum),order+1);
for i = 0:order
   Jx3 = besselj(i,k3r);
   Nx3 = bessely(i,k3r);
   H = Jx3'-im*Nx3';
    if i==0
        sum_terms(:,i+1) = sin(eta(:,i+1)).*exp(im.*eta(:,i+1)).*H.*(-im)^(i+1).*cos(i*theta);
        sum = sum+sin(eta(:,i+1)).*exp(im.*eta(:,i+1)).*H.*(-im)^(i+1).*cos(i*theta);
    else
        sum_terms(:,i+1) = 2.*sin(eta(:,i+1)).*exp(im.*eta(:,i+1)).*H.*(-im)^(i+1).*cos(i*theta);
        sum = sum+2.*sin(eta(:,i+1)).*exp(im.*eta(:,i+1)).*H.*(-im)^(i+1).*cos(i*theta);
    end
end

sum(isnan(sum)) = 0;
sum = conj(sum);
inc_f = fft(tone_burst, Nf);
back_f = inc_f.*sum'.*(-1);
back_f(isnan(back_f)) = 0;
back = real(ifft(back_f));
back = back(1,1:length(tone_burst));
end