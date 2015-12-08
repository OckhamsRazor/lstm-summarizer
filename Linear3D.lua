local Linear3D,parent = torch.class('nn.Linear3D', 'nn.Linear')

function Linear3D:__init(inputSize, outputSize)
  parent.__init(self, inputSize, outputSize)

  self.dim1 = inputSize
  self.dim2 = outputSize
end

function Linear3D:updateOutput(input)
  self.bs, self.nseq, _ = table.unpack(input:size():totable())

  self.output:resize(self.bs * self.nseq, self.dim2)
  parent.updateOutput(self,input:view(-1, self.dim1))
  self.output = self.output:view(self.bs, self.nseq, -1)
  return self.output
end

function Linear3D:updateGradInput(input, gradOutput)
  self.gradInput:resize(self.bs * self.nseq, self.dim1)
  parent.updateGradInput(self, input:view(-1, self.dim1), gradOutput:view(-1,self.dim2))
  self.gradInput = self.gradInput:view(self.bs, self.nseq, -1)
  return self.gradInput
end
