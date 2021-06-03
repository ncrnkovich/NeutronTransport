%% neutronTransport

a = 1;
I = 100;
N = 8;
xspace = linspace(0, a, I);
delta = xspace(3) - xspace(1);
psi = zeros(N, I);
% vacuum boundary conditions on both sides
psi(:,1) = 0; psi(:,end) = 0;

sig = 2.74e-24; % barns to cm^2
sigV = zeros(1, length(psi));
sigV(1:end) = sig;

% P_N quadrature; N = 8
mu_n = [-0.9602898564,-0.7966664774,-0.5255324099,-0.18343464240,0.1834346424,0.52553240990,0.7966664774,0.9602898564];
w_n = [-0.1012285363,-0.2223810344,-0.3137066459,-0.3626837834,0.3626837834,0.3137066459,0.2223810344,0.1012285363];

phi = zeros(1, I);
phiPrev = zeros(1,length(phi));
error = 10;
err = 10^-9;
S = 10e-5; % what do you set this source to?
q = zeros(1, I);
q(1:end) = sig/2*phi0 + S;

while error > err
    
    for n = 1:length(mu_n)
        
        if mu_n(n) > 0
            
            for i = 2:2:I-1
                
                psi(n,i) = (1 + 0.5*sigV(i)*delta/abs(mu_n(n)))^(-1)*(psi(n,i-1) + 0.5*delta*q(i)/abs(mu_n(n)));
                psi(n,i+1) = 2*psi(n,i) - psi(n,i-1);
            end
            
        else
            for i = I-1:-2:2
                
                psi(n,i) = (1 + 0.5*sigV(i)*delta/abs(mu_n(n)))^(-1)*(psi(n,i+1) + 0.5*delta*q(i)/abs(mu_n(n)));
                psi(n,i-1) = 2*psi(n,i) - psi(n,i+1);
            end
        end
        
    end
    
    for i = 2:2:I-1
        phi(i) = dot(w_n, psi(:,i));
        q(i) = 0.5*sigV(i)*phi(i) + S;
    end
    
    error = max(abs(phiPrev - phi));
    phiPrev = phi;
    
end
    

    
    



