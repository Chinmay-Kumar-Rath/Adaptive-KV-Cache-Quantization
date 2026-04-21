import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

class KVCache:
    def __init__(self):
        self.K = None
        self.V = None
        self.importance = []

    def add(self, k, v):
        if self.K is None:
            self.K = k.to(device)
            self.V = v.to(device)
        else:
            self.K = torch.cat([self.K, k], dim=0)
            self.V = torch.cat([self.V, v], dim=0)
            """dim=0 means (1,4) + (1,4) → (2,4)
                    Step 1:
                    [ K1 ]

                    Step 2:
                    [ K1
                    K2 ]

                    Step 3:
                    [ K1
                    K2
                    K3 ]
               dim=1 means (1,4) + (1,4) → (1,8) """
        self.importance.append(0.0)

    def importance_update(self, weights,decay=0.95):
        for i in range(len(self.importance)):
            if i<4:                  # first 4 tokens accumulate importance without decay
                self.importance[i]+=weights[
                    0, i
                ].item()  # item() converts tensor(0.6)->0.6 also 0 in [0,i] means zero inexed row (only 1 row(1 token) at a time)
            else:
                    self.importance[i] = (decay * self.importance[i])+weights[
                    0, i
                ].item() 

    def get(self):
        return self.K, self.V
