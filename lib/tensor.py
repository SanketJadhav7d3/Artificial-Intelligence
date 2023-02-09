
import numpy as np

class Tensor(object):

    def __init__(self, data, autograd=False, creators=None, creation_op=None, id=None):
        self.data = np.array(data)
        self.creators = creators
        self.creation_op = creation_op
        self.grad = None
        self.autograd = autograd
        self.children = {}

        if id is None:
            id = np.random.randint(0, 100000)
        self.id = id

        if creators is not None:
            for c in creators:
                if self.id not in c.children:
                    c.children[self.id] = 1
                else:
                    c.children[self.id] += 1

    def all_children_grads_accounted_for(self):
        for id, cnt in self.children.items():
            if (cnt != 0):
                return False
        return True

    def backward(self, grad=None, grad_origin=None):
        if self.autograd:
            if grad_origin is not None:
                if (self.children[grad_origin.id] == 0):
                    raise Exception("Cannot backpropogate more than once")
                else:
                    self.children[grad_origin.id] -= 1

            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad

            if self.creators is not None and (self.all_children_grads_accounted_for() 
                                              or grad_origin is None):
                if self.creation_op == "add":
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)

                if self.creation_op == "neg":
                    self.creators[0].backward(self.grad.__neg__())

                if self.creation_op == "sub":
                    new = Tensor(self.grad.data)
                    self.creators[0].backward(new, self)
                    new = Tensor(self.grad.__neg__().data)
                    self.creators[1].backward(new, self)

                if self.creation_op == "mul":
                    new = self.grad * self.creators[1]
                    self.creators[0].backward(new ,self)
                    new = self.grad * self.creators[0]
                    self.creators.backward(new, self)

                if self.creation_op == "mm":
                    act = self.creators[0]
                    weights = self.creators[1]
                    new = self.grad.mm(weights.transpose())
                    act.backward(new)
                    new = self.grad.transpose().mm(act).transpose()
                    weights.backward()

                if(self.creation_op == "transpose"): 
                    self.creators[0].backward(self.grad.transpose())

                if("sum" in self.creation_op):
                    dim = int(self.creation_op.split("_")[1])
                    ds = self.creators[0].data.shape[dim] 
                    self.creators[0].backward(self.grad.expand(dim,ds))

                if("expand" in self.creation_op):
                    dim = int(self.creation_op.split("_")[1]) 
                    self.creators[0].backward(self.grad.sum(dim))

    def __add__(self, other):
        if (self.autograd and other.autograd):
            return Tensor(self.data + other.data, 
                          autograd=True, 
                          creators=[self, other],
                          creation_op="add")

        return Tensor(self.data + other.data)

    def __neg__(self):
        if self.autograd:
            return Tensor(self.data * -1,
                          autograd=True,
                          creators=[self],
                          creation_op="neg")
        return Tensor(self.data * -1)

    def __sub__(self, other):
        if self.autograd:
            return Tensor(self.data - other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op="sub")
        return Tensor(self.data - other.data)

    def __mul__(self, other):
        if self.autograd:
            return Tensor(self.data * other.data, 
                          autograd=True,
                          creators=[self, other],
                          creation_op="mul")
        return Tensor(self.data * other.data)

    def sum(self, dim):
        if self.autograd:
            return Tensor(self.data.sum(dim),
                          autograd=True,
                          creators=[self],
                          creation_op="sum_"+str(dim))
        return Tensor(self.data.sum(dim))

    def expand(self, dim, copies):
        trans_cmd = list(range(0,len(self.data.shape))) 
        trans_cmd.insert(dim,len(self.data.shape))
        new_shape = list(self.data.shape) + [copies]
        new_data = self.data.repeat(copies).reshape(new_shape) 
        new_data = new_data.transpose(trans_cmd)

        if(self.autograd):
            return Tensor(new_data,
                          autograd=True,
                          creators=[self], 
                          creation_op="expand_"+str(dim))

        return Tensor(new_data)

    def transpose(self):
        if self.autograd:
            return Tensor(self.data.transpose(), 
                          autograd=True,
                          creators=[self],
                          creation_op="transpose")
        return Tensor(self.data.tranpose())

    def mm(self, other):
        if self.autograd:
            return Tensor(self.data.dot(other.data),
                          autograd=True,
                          creators=[self, other],
                          creation_op="mm")

        return Tensor(self.data.dot(other.data))

    def __repr__(self):
        return "Tensor: " + str(self.data.__repr__())
