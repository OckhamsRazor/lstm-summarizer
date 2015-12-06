local cmd = torch.CmdLine()
cmd:option('-gpuidx', 1, 'Index of GPU on which job should be executed.')
cmd:option('-max_seq_length', 30, 'Maximum input document length (in sentences).')
cmd:option('-batch_size', 32, 'Document batch size.')
cmd:option('-train', 'train', 'Training data folder path.')
cmd:option('-valid', 'valid', 'Validation data folder path.')
cmd:option('-test', 'test', 'Testing data folder path.')
cmd:option('-enc', 'model/enc.net', 'Sentence encoder path.')
local opt = cmd:parse(arg)

local ok,cunn = pcall(require, 'fbcunn')
if not ok then
    ok,cunn = pcall(require,'cunn')
    if ok then
        print("warning: fbcunn not found. Falling back to cunn")
        LookupTable = nn.LookupTable
    else
        print("Could not find cunn or fbcunn. Either is required")
        os.exit()
    end
else
    deviceParams = cutorch.getDeviceProperties(1)
    cudaComputeCapability = deviceParams.major + deviceParams.minor/10
    LookupTable = nn.LookupTable
end
require('paths')
require('nngraph')
require('base')

local function transfer_data(x)
  return x:cuda()
end

-- Train 1 day and gives 82 perplexity.
--[[
local params = {batch_size=20,
                max_seq_length=tonumber(arg[2]),
                layers=2,
                decay=1.15,
                rnn_size=1000,
                dropout=0.65,
                init_weight=0.08,
                lr=0.1,
                vocab_size=25002,
                max_epoch=14,
                max_max_epoch=55,
                max_grad_norm=10}

--]]
-- Trains 1h and gives test 115 perplexity.
-- [[
local params_sent = {batch_size=1,
                     max_seq_length=30,
                     layers=2,
                     rnn_size=1000}

local params = {batch_size=tonumber(opt.batch_size),
                max_seq_length=tonumber(opt.max_seq_length),
                layers=2,
                decay=2,
                rnn_size=1000,
                dropout=0.2,
                init_weight=0.08,
                lr=0.1,
                vocab_size=25002,
                max_epoch=4,
                max_max_epoch=13,
                max_grad_norm=5}

local word_emb_size = 2*params_sent.layers*params_sent.rnn_size
local stringx = require('pl.stringx')
local EOS = params.vocab_size-1
local NIL = params.vocab_size

function load_data_into_doc(fname)
  local doc = {}
  local content = {}

  local file = torch.DiskFile(fname, 'r')
  file:quiet()
  local title = file:readString('*l')
  title = string.gsub(title, '%s*$', '')
  title = stringx.split(title)
  table.insert(title, EOS)
  for wid = #title+1, params_sent.max_seq_length do
    table.insert(title, NIL)
  end

  file:readString('*l')
  repeat
    local sent = file:readString('*l')
    sent = string.gsub(sent, '%s*$', '')
    if #sent ~= 0 then
      sent = stringx.split(sent)
      table.insert(sent, EOS)
      for wid = #sent+1, params_sent.max_seq_length do
        table.insert(sent, NIL)
      end
      table.insert(content, sent)
    -- else
      -- table.insert(docs, doc)
      -- doc = {}
    end
  until file:hasError()

  table.insert(doc, content)
  table.insert(doc, title)
  return doc
end

local state_train, state_valid, state_test
local model = {}
local paramx_enc, paramdx_enc, paramx_dec, paramdx_dec

local function lstm(x, prev_c, prev_h)
  -- Calculate all four gates in one go
  local i2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(x)
  local h2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(prev_h)
  local gates = nn.CAddTable()({i2h, h2h})

  -- Reshape to (batch_size, n_gates, hid_size)
  -- Then slize the n_gates dimension, i.e dimension 2
  local reshaped_gates =  nn.Reshape(4,params.rnn_size)(gates)
  local sliced_gates = nn.SplitTable(2)(reshaped_gates)

  -- Use select gate to fetch each gate and apply nonlinearity
  local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
  local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
  local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
  local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
  })
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

  return next_c, next_h
end

local function create_network()
  local x                = nn.Identity()()
  local y                = nn.Identity()()
  local prev_s_enc       = nn.Identity()()
  local i_0              = nn.Linear(word_emb_size, params.rnn_size)(x)
  local i                = {[0] = i_0}
  local next_s_enc       = {}
  local split_enc        = {prev_s_enc:split(2 * params.layers)}
  for layer_idx = 1, params.layers do
    local prev_c         = split_enc[2 * layer_idx - 1]
    local prev_h         = split_enc[2 * layer_idx]
    local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1])
    local next_c, next_h = lstm(dropped, prev_c, prev_h)
    table.insert(next_s_enc, next_c)
    table.insert(next_s_enc, next_h)
    i[layer_idx] = next_h
  end

  local encoder = nn.gModule(
    {x, prev_s_enc},
    {nn.Identity()(next_s_enc)}
  )

  local prev_s_dec      = nn.Identity()()
  local j_0             = nn.Identity()()
  local j               = {[0] = j_0}
  local next_s_dec      = {}
  local split_dec       = {prev_s_dec:split(2 * params.layers)}
  for layer_idx = 1, params.layers do
    local prev_c         = split_dec[2 * layer_idx - 1]
    local prev_h         = split_dec[2 * layer_idx]
    local dropped        = nn.Dropout(params.dropout)(j[layer_idx - 1])
    local next_c, next_h = lstm(dropped, prev_c, prev_h)
    table.insert(next_s_dec, next_c)
    table.insert(next_s_dec, next_h)
    j[layer_idx] = next_h
  end

  local h2y              = nn.Linear(params.rnn_size, params.vocab_size)
  local dropped          = nn.Dropout(params.dropout)(j[params.layers])
  local pred             = nn.LogSoftMax()(h2y(dropped))
  local mask             = torch.ones(params.vocab_size)
  mask[NIL] = 0
  local err              = nn.ClassNLLCriterion(mask)({pred, y})

  local decoder = nn.gModule(
    {j_0, y, prev_s_dec},
    {err, nn.Identity()(next_s_dec), pred}
  )

  encoder:getParameters():uniform(-params.init_weight, params.init_weight)
  decoder:getParameters():uniform(-params.init_weight, params.init_weight)
  encoder = transfer_data(encoder)
  decoder = transfer_data(decoder)

  return {encoder, decoder}
end

local function setup()
  print("Creating a RNN LSTM network.")
  local sent_encoder = torch.load(opt.enc)
  local encoder, decoder = unpack(create_network())
  paramx_enc, paramdx_enc = encoder:getParameters()
  paramx_dec, paramdx_dec = decoder:getParameters()

  model.s_sent_enc = {}
  model.start_s_sent_enc = {}

  for j = 0, params_sent.max_seq_length do
    model.s_sent_enc[j] = {}
    for d = 1, 2 * params_sent.layers do
      model.s_sent_enc[j][d] = transfer_data(torch.zeros(params_sent.batch_size, params_sent.rnn_size))
    end
  end
  for d = 1, 2 * params_sent.layers do
    model.start_s_sent_enc[d] = transfer_data(torch.zeros(params_sent.batch_size, params_sent.rnn_size))
  end

  model.s_enc = {}
  model.ds_enc = {}
  model.start_s_enc = {}
  model.s_dec = {}
  model.ds_dec = {}
  model.start_s_dec = {}

  model.preds = {}

  for j = 0, params.max_seq_length do
    model.s_enc[j] = {}
    model.s_dec[j] = {}
    for d = 1, 2 * params.layers do
      model.s_enc[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
      model.s_dec[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end
  end
  for d = 1, 2 * params.layers do
    model.start_s_enc[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    model.ds_enc[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    model.start_s_dec[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    model.ds_dec[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  end

  model.sent_encoder = sent_encoder
  model.encoder = encoder
  model.decoder = decoder
  model.rnns_sent_enc = g_cloneManyTimes(sent_encoder, params_sent.max_seq_length)
  model.rnns_enc = g_cloneManyTimes(encoder, params.max_seq_length)
  model.rnns_dec = g_cloneManyTimes(decoder, params.max_seq_length)
  model.norm_dw_enc = 0
  model.norm_dw_dec = 0
  model.err = transfer_data(torch.zeros(params.max_seq_length))
end

local function reset_state(state)
  state.pos = 0
  if model ~= nil
    and model.start_s_enc ~= nil
    and model.start_s_dec ~= nil then
    for d = 1, 2 * params.layers do
      model.start_s_enc[d]:zero()
      model.start_s_dec[d]:zero()
    end
  end
end

local function reset_ds()
  for d = 1, #model.ds_enc do
    model.ds_enc[d]:zero()
    model.ds_dec[d]:zero()
  end
end

local function get_embeddings(sents, batch_size)
  local embs
  -- pad document if too short
  for i = #sents+1, batch_size do
    table.insert(sents, torch.ones(params.max_seq_length):mul(NIL))
  end

  for idx, sent in pairs(sents) do
    -- stop if document too long
    if idx > batch_size then break end

    -- reset state of sentence encoder
    for i = 1, 2*params_sent.layers do
      model.s_sent_enc[0][i]:zero()
    end

    -- concate sentence if too long
    local data = torch.Tensor(sent)
    if data:size()[1] > params_sent.max_seq_length then
      data = data[{{1,params_sent.max_seq_length}}]
    end

    -- make sentence a cuda tensor
    data:resize(params_sent.max_seq_length, 1)
    data = transfer_data(data)

    -- encode it!
    local eos_pos = params_sent.max_seq_length
    for i = 1, params_sent.max_seq_length do
      local x = data[i]
      local s_sent_enc = model.s_sent_enc[i - 1]
      _, model.s_sent_enc[i] = unpack(
        model.rnns_sent_enc[i]:forward({x, s_sent_enc})
      )

      -- if sentence reaches <eos> at i-th word...
      if x == EOS then eos_pos = i end
    end

    -- copy the result to ret
    local emb = transfer_data(torch.zeros(1, word_emb_size))
    for l = 1, 2*params_sent.layers do
      local start_id = (l-1)*params_sent.rnn_size+1
      local end_id   =  l   *params_sent.rnn_size
      emb[1][{{start_id, end_id}}]:copy(model.s_sent_enc[eos_pos][l][1])
    end

    if embs == nil then
      embs = emb:float()
    else
      embs = torch.cat(embs, emb:float(), 1)
    end
  end

  return embs
end

local function _fp(state)
  g_replace_table(model.s_enc[0], model.start_s_enc)

  -- if state.pos + params.batch_size > #state.data then
  --   reset_state(state)
  -- end

  local x_batch, y_batch
  for data_id = 1, params.batch_size do
    local data = state.data[state.pos+data_id]
    local x_embs = get_embeddings(data[1], params.max_seq_length)
    -- local y_embs = get_embeddings({data[2]}, 1)
    local ys = torch.Tensor(data[2]):resize(params_sent.max_seq_length, 1)
    x_embs:resize(params.max_seq_length, 1, word_emb_size)
    if x_batch == nil then
      x_batch = x_embs
      y_batch = ys
    else
      x_batch = torch.cat(x_batch, x_embs, 2)
      y_batch = torch.cat(y_batch, ys, 2)
    end
  end
  x_batch = transfer_data(x_batch)
  y_batch = transfer_data(y_batch)
  state.x_batch = x_batch
  state.y_batch = y_batch

  for i = 1, params.max_seq_length do
    local x = state.x_batch[i]
    local s_enc = model.s_enc[i-1]
    model.s_enc[i] = model.rnns_enc[i]:forward({x, s_enc})
  end

  g_replace_table(model.s_dec[0], model.s_enc[params.max_seq_length])

  for i = 1, params_sent.max_seq_length do
    local y = state.y_batch[i]
    local s_dec = model.s_dec[i-1]
    local emb = s_dec[2*params.layers]
    model.err[i], model.s_dec[i] = unpack(
      model.rnns_dec[i]:forward({emb, y, s_dec})
    )
  end

  g_replace_table(model.start_s_enc, model.s_enc[params.max_seq_length])
  return model.err:mean()
end

local function _bp(state) -- TODO from here
  paramdx_enc:zero()
  paramdx_dec:zero()
  reset_ds()

  for i = params_sent.max_seq_length, 1, -1 do
    local y = state.y_batch[i]
    local s_dec = model.s_dec[i-1]
    local embeddings = s_dec[2*params.layers]
    local d_pred = transfer_data(
      torch.zeros(params.batch_size, params.vocab_size)
    )
    local derr = transfer_data(torch.ones(1))

    local tmp_dec
    tmp_dec = model.rnns_dec[i]:backward(
      {embeddings, y, s_dec},
      {derr, model.ds_dec, d_pred}
    )[3]

    g_replace_table(model.ds_dec, tmp_dec)
    cutorch.synchronize()
  end

  g_replace_table(model.ds_enc, model.ds_dec)

  for i = params.max_seq_length, 1, -1 do
    local x = state.x_batch[i]
    local s_enc = model.s_enc[i-1]
    local tmp_enc = model.rnns_enc[i]:backward(
      {x, s_enc}, model.ds_enc
    )[2]

    g_replace_table(model.ds_enc, tmp_enc)
    cutorch.synchronize()
  end

  -- state.pos = state.pos + params.batch_size
  model.norm_dw_enc = paramdx_enc:norm()
  model.norm_dw_dec = paramdx_dec:norm()
  if model.norm_dw_enc > params.max_grad_norm then
    local shrink_factor = params.max_grad_norm / model.norm_dw_enc
    paramdx_enc:mul(shrink_factor)
  end
  if model.norm_dw_dec > params.max_grad_norm then
    local shrink_factor = params.max_grad_norm / model.norm_dw_dec
    paramdx_dec:mul(shrink_factor)
  end
  paramx_enc:add(paramdx_enc:mul(-params.lr))
  paramx_dec:add(paramdx_dec:mul(-params.lr))
end

local function run_valid()
  reset_state(state_valid)
  g_disable_dropout(model.rnns_enc)
  g_disable_dropout(model.rnns_dec)
  local len = #state_valid.data / params.batch_size
  local perp = 0
  for i = 1, len do
    perp = perp + _fp(state_valid)
    state_valid.pos = state_valid.pos + params.batch_size
  end
  print("Validation set perplexity : " .. g_f3(torch.exp(perp / len)))
  g_enable_dropout(model.rnns_enc)
  g_enable_dropout(model.rnns_dec)
end

local function run_test()
  reset_state(state_test)
  g_disable_dropout(model.rnns_enc)
  g_disable_dropout(model.rnns_dec)
  local len = #state_test.data / params.batch_size
  local perp = 0
  for i = 1, len do
    perp = perp + _fp(state_test)
    state_test.pos = state_test.pos + params.batch_size
  end
  print("Test set perplexity : " .. g_f3(torch.exp(perp / len)))
  g_enable_dropout(model.rnns_enc)
  g_enable_dropout(model.rnns_dec)
end

local function main()
  g_init_gpu(opt.gpuidx)

  local train_f = {}
  local valid_f = {}
  local test_f = {}
  for f in paths.iterfiles(opt.train) do
    table.insert(train_f, f)
  end
  for f in paths.iterfiles(opt.valid) do
    table.insert(valid_f, f)
  end
  for f in paths.iterfiles(opt.test) do
    table.insert(test_f, f)
  end
  local num_of_docs = #train_f
  local epoch_size = torch.floor(num_of_docs / params.batch_size)

  local train_docs = {}
  local valid_docs = {}
  local test_docs = {}
  -- for _, f in pairs(train_f) do
    -- local doc = load_data_into_doc(paths.concat(opt.train, f))
    -- table.insert(train_docs, doc)
    -- if #train_docs == params.batch_size then break end
  -- end
  -- for _, f in pairs(valid_f) do
  --   local doc = load_data_into_doc(paths.concat(opt.valid, f))
  --   table.insert(valid_docs, doc)
  -- end
  -- for _, f in pairs(test_f) do
  --   local doc = load_data_into_doc(paths.concat(opt.test, f))
  --   table.insert(test_docs, doc)
  -- end
  state_train = {data=train_docs}
  state_valid = {data=valid_docs}
  state_test = {data=test_docs}
  setup()
  reset_state(state_train)

  local step = 0
  local epoch = 0
  local total_cases = 0
  local beginning_time = torch.tic()
  local start_time = torch.tic()
  print("Starting training.")

  local perps
  local file_step = 0
  -- local file_size = torch.floor(#state_train.data / params.batch_size)
  while epoch < params.max_max_epoch do
    state_train.data = {}
    for fid = 1, params.batch_size do
      local f = train_f[file_step+fid]
      local doc = load_data_into_doc(paths.concat(opt.train, f))
      table.insert(state_train.data, doc)
    end
    local perp = _fp(state_train)
    print(perp)

    if perps == nil then
      perps = torch.zeros(epoch_size):add(perp)
    end
    perps[step % epoch_size + 1] = perp
    step = step + 1
    file_step = file_step + params.batch_size
    _bp(state_train)

    total_cases = total_cases + params.max_seq_length * params.batch_size
    epoch = step / epoch_size
    if step % torch.round(epoch_size / 10) == 10 then
      local wps = torch.floor(total_cases / torch.toc(start_time))
      local since_beginning = g_d(torch.toc(beginning_time) / 60)
      print('epoch = ' .. g_f3(epoch) ..
            ', train perp. = ' .. g_f3(torch.exp(perps:mean())) ..
            ', wps = ' .. wps ..
            ', encoder dw:norm() = ' .. g_f3(model.norm_dw_enc) ..
            ', decoder dw:norm() = ' .. g_f3(model.norm_dw_dec) ..
            ', lr = ' ..  g_f3(params.lr) ..
            ', since beginning = ' .. since_beginning .. ' mins.')
    end
    if step % epoch_size == 0 then
      run_valid()
      if epoch > params.max_epoch then
          params.lr = params.lr / params.decay
      end

      torch.save(
        'models/'..tostring(torch.floor(epoch))..'.enc', model.encoder
      )
      torch.save(
        'models/'..tostring(torch.floor(epoch))..'.dec', model.decoder
      )
    end
    if step % 33 == 0 then
      cutorch.synchronize()
      collectgarbage()
    end

    -- if step == file_size then
    --   file_step = file_step + 1
    --   if file_step > #filenames then
    --     file_step = 1
    --   end

    --   print('Current file: ' .. filenames[file_step] ..
    --         ', file no. ' .. file_step ..
    --         ', current step: ' .. step)

    --   state_train.data = load_data_into_sents(
    --     paths.concat(opt.train, filenames[file_step])
    --   )
    --   state_train.pos = 1
    --   file_size = file_size + torch.floor(
    --     #state_train.data / params.batch_size
    --   )
    -- end
  end
  run_test()
  print("Training is over.")
end

main()
