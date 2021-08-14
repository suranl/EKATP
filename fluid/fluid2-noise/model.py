from torch import nn
import torch

P=22
def gaussian_init_(n_units, std=1):    
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]  
    return Omega


class encoderNet(nn.Module):
    def __init__(self, m, n, b, ALPHA = 1):
        super(encoderNet, self).__init__()
        self.N = m * n
        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(self.N, P*ALPHA)
        self.fc2 = nn.Linear(P*ALPHA, P*ALPHA)
        self.fc3 = nn.Linear(P*ALPHA, b)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)          

    def forward(self, x):
        x = x.view(-1, 1, self.N)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        
        return x


class decoderNet(nn.Module):
    def __init__(self, m, n, b, ALPHA = 1):
        super(decoderNet, self).__init__()

        self.m = m
        self.n = n
        self.b = b

        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(b, P*ALPHA)
        self.fc2 = nn.Linear(P*ALPHA, P*ALPHA)
        self.fc3 = nn.Linear(P*ALPHA, m*n)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)          

    def forward(self, x):
        x = x.view(-1, 1, self.b)
        x = self.tanh(self.fc1(x)) 
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, 1, self.m, self.n)
        return x



class dynamicsC(nn.Module):
    def __init__(self, b, init_scale):
        super(dynamicsC, self).__init__()

        self.dynamics = nn.Linear(b, b, bias=False)
        self.fixed = nn.Linear(b, b-1, bias=False)
        for p in self.parameters():
            p.requires_grad=False
        self.flexi = nn.Linear(b, 1, bias=False)

        for j in range(0,b):
            self.dynamics.weight.data[b-1][j]=self.flexi.weight.data[0][j]=0
        self.dynamics.weight.data[b-1][0]=1

        for i in range(0,b-1):
            for j in range (0,b):
                if i+1==j:
                    self.dynamics.weight.data[i][j]=1
                    self.fixed.weight.data[i][j]=1
                else:
                    self.dynamics.weight.data[i][j]=0
                    self.fixed.weight.data[i][j]=0

        print('----')
        print(self.dynamics.weight)
        #print(self.fixed.weight)
        #print(self.flexi.weight)
        self.tanh = nn.Tanh()

    def forward(self, x):
        up = self.fixed(x)
        down = self.flexi(x)
        x = torch.cat((up,down),2)

        self.dynamics.weight.data = torch.cat(( self.fixed.weight.data,self.flexi.weight.data),0)
        #print("self.dynamics.weight.data=",self.dynamics.weight.data)
        return x


class dynamics_backD(nn.Module):
    def __init__(self, b, omega):
        super(dynamics_backD, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.fixed = nn.Linear(b, b-1, bias=False)
        for p in self.parameters():
            p.requires_grad=False

        self.flexi = nn.Linear(b, 1, bias=False)


        for j in range(0,b-1):
            self.dynamics.weight.data[0][j]=-omega.dynamics.weight.data[b-1][j+1]/omega.dynamics.weight.data[b-1][0]
            self.flexi.weight.data[0][j]=self.dynamics.weight.data[0][j]
        self.flexi.weight.data[0][b-1]=self.dynamics.weight.data[0][b-1]=1.0/omega.dynamics.weight.data[b-1][0]

        for i in range(1,b):
            for j in range (0,b):
                if i-1==j:
                    self.dynamics.weight.data[i][j]=1
                    self.fixed.weight.data[i-1][j]=1
                else:
                    self.dynamics.weight.data[i][j]=0
                    self.fixed.weight.data[i-1][j]=0

        #print(self.dynamics.weight)
        #print(self.flexi.weight)


    def forward(self, x):
        up = self.flexi(x)
        down = self.fixed(x)
        x = torch.cat((up,down),2)
        self.dynamics.weight.data = torch.cat(( self.flexi.weight.data,self.fixed.weight.data),0)
        return x


class embed_koopmanAE(nn.Module):
    def __init__(self, m, n, b, steps, steps_back, alpha = 1, init_scale=1):
        super(embed_koopmanAE, self).__init__()
        self.steps = steps
        self.steps_back = steps_back
        
        self.encoder = encoderNet(m, n, b, ALPHA = alpha)
        self.dynamics = dynamicsC(b, init_scale)
        self.backdynamics = dynamics_backD(b, self.dynamics)
        self.decoder = decoderNet(m, n, b, ALPHA = alpha)


    def forward(self, x, mode='forward'):
        out = []
        out_back = []
        z = self.encoder(x.contiguous())
        q = z.contiguous()

        
        if mode == 'forward':
            for _ in range(self.steps):
                q = self.dynamics(q)
                out.append(self.decoder(q))

            out.append(self.decoder(z.contiguous())) 
            return out, out_back    

        if mode == 'backward':
            for _ in range(self.steps_back):
                q = self.backdynamics(q)
                out_back.append(self.decoder(q))
                
            out_back.append(self.decoder(z.contiguous()))
            return out, out_back