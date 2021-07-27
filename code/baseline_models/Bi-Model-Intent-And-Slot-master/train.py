from torch import optim
import numpy as np
import torch

import utils
from utils import get_chunks
from config import device
import config as cfg
from data2index_ver2 import train_data, test_data, index2slot_dict
from model import *
import pickle
from sklearn.metrics import accuracy_score
epoch_num = cfg.total_epoch

slot_model = Slot().to(device)
intent_model = Intent().to(device)

print(slot_model)
print(intent_model)

slot_optimizer = optim.Adam(slot_model.parameters(), lr=cfg.learning_rate)       # optim.Adamax
intent_optimizer = optim.Adam(intent_model.parameters(), lr=cfg.learning_rate)   # optim.Adamax

best_correct_num = 0
best_epoch = -1
best_F1_score = 0.0
best_epoch_slot = -1
for epoch in range(epoch_num):
    slot_loss_history = []
    intent_loss_history = []
    for batch_index, data in enumerate(utils.get_batch(train_data)):

	    # Preparing data
        sentence, real_len, slot_label, intent_label = data

        mask = utils.make_mask(real_len, batch=len(sentence)).to(device)
        x = torch.tensor(sentence).to(device)
        y_slot = torch.tensor(slot_label).to(device)
        y_slot = utils.one_hot(y_slot).to(device)
        y_intent = torch.tensor(intent_label).to(device)
        # y_intent = utils.one_hot(y_intent, Num=18).to(device)
        y_intent = utils.one_hot(y_intent, Num=4).to(device)

		# Calculate compute graph
        slot_optimizer.zero_grad()
        intent_optimizer.zero_grad()
		
        hs = slot_model.enc(x)
        slot_model.share_memory = hs.clone()

        hi = intent_model.enc(x)
        intent_model.share_memory = hi.clone()
		
		
        slot_logits = slot_model.dec(hs, intent_model.share_memory.detach())
        log_slot_logits = utils.masked_log_softmax(slot_logits, mask, dim=-1)
        slot_loss = -1.0*torch.sum(y_slot*log_slot_logits)
        slot_loss_history.append(slot_loss.item())
        slot_loss.backward()
        torch.nn.utils.clip_grad_norm_(slot_model.parameters(), 5.0)
        slot_optimizer.step()

        # Asynchronous training
        intent_logits = intent_model.dec(hi, slot_model.share_memory.detach(), real_len)
        log_intent_logits = F.log_softmax(intent_logits, dim=-1)
        intent_loss = -1.0*torch.sum(y_intent*log_intent_logits)
        intent_loss_history.append(intent_loss.item())
        intent_loss.backward()
        torch.nn.utils.clip_grad_norm_(intent_model.parameters(), 5.0)
        intent_optimizer.step()
        # break
		# Log
        if batch_index % 100 == 0 and batch_index > 0:
            print('Slot loss: {:.4f} \t Intent loss: {:.4f}'.format(sum(slot_loss_history[-100:])/100.0, \
                sum(intent_loss_history[-100:])/100.0))

    # Evaluation 
    total_test = len(test_data)
    correct_num = 0
    TP, FP, FN = 0, 0, 0

    test_pred_intent = []
    test_correct_intent = []
    test_pred_slot = []
    test_correct_slot = []
    
    for batch_index, data_test in enumerate(utils.get_batch(test_data, batch_size=32)):
        sentence_test, real_len_test, slot_label_test, intent_label_test = data_test
        # print(sentence[0].shape, real_len.shape, slot_label.shape)
        # print(len(sentence_test))
        # print(len(intent_label_test))
        x_test = torch.tensor(sentence_test).to(device)

        # mask_test = utils.make_mask(real_len_test, batch=32).to(device)
        mask_test = utils.make_mask(real_len_test, batch=len(sentence_test)).to(device)

        # Slot model generate hs_test and intent model generate hi_test
        hs_test = slot_model.enc(x_test)
        hi_test = intent_model.enc(x_test)

        # Slot
        slot_logits_test = slot_model.dec(hs_test, hi_test)
        log_slot_logits_test = utils.masked_log_softmax(slot_logits_test, mask_test, dim=-1)
        slot_pred_test = torch.argmax(log_slot_logits_test, dim=-1)
        # Intent
        intent_logits_test = intent_model.dec(hi_test, hs_test, real_len_test)
        log_intent_logits_test = F.log_softmax(intent_logits_test, dim=-1)
        res_test = torch.argmax(log_intent_logits_test, dim=-1)
        
        test_pred_intent.extend(res_test.tolist())
        test_correct_intent.extend(intent_label_test)


      #   if res_test.item() == intent_label_test[0]:
      #       correct_num += 1
      #   if correct_num > best_correct_num:
      #       best_correct_num = correct_num
      #       best_epoch = epoch
			# # Save and load the entire model.
      #       torch.save(intent_model, 'model_intent_best.ckpt')
      #       torch.save(slot_model, 'model_slot_best.ckpt')
    
        # Calc slot F1 score
        assert len(slot_pred_test) == len(slot_label_test)
        assert len(slot_pred_test) == len(real_len_test)
        assert len(test_pred_intent) == len(test_pred_intent)
        slot_pred_test_new = []
        slot_label_test_new = []
        for i in range(len(slot_pred_test)):
          pred = slot_pred_test[i]
          slot_label = slot_label_test[i]
          slot_pred_test_new.append(pred[:real_len_test[i]])
          slot_label_test_new.append(slot_label[:real_len_test[i]])
        # slot_pred_test = slot_pred_test[0][:real_len_test[0]]
        # slot_label_test = slot_label_test[0][:real_len_test[0]]
        # print('slot_pred_test:',slot_pred_test)
        # print('slot_label_test:',slot_label_test)
        slot_pred_test = [[int(item_in) for item_in in item] for item in slot_pred_test_new]
        slot_label_test = [[int(item_in) for item_in in item] for item in slot_label_test_new]
        # print('slot_pred_test:',slot_pred_test)
        # print('slot_label_test:',slot_label_test)
        slot_pred_test = [[index2slot_dict[item_in] for item_in in item] for item in slot_pred_test]
        slot_label_test = [[index2slot_dict[item_in] for item_in in item] for item in slot_label_test]

        # pred_chunks = get_chunks(['O'] + slot_pred_test + ['O'])
        # label_chunks = get_chunks(['O'] + slot_label_test + ['O'])
        # for pred_chunk in pred_chunks:
        #     if pred_chunk in label_chunks:
        #         TP += 1
        #     else:
        #         FP += 1
        # for label_chunk in label_chunks:
        #     if label_chunk not in pred_chunks:
        #         FN += 1
        
        test_pred_slot.extend(slot_pred_test)
        test_correct_slot.extend(slot_label_test)

    print('test_pred_intent:',len(test_pred_intent))
    print('test_correct_intent:',len(test_correct_intent))
    print('test_pred_slot:',len(test_pred_slot))
    print('test_correct_slot:',len(test_correct_slot))

    pickle.dump(test_pred_intent, open( "./results2/test_pred_intent"+str(epoch)+".pkl", "wb" ) )
    pickle.dump(test_correct_intent, open( "./results2/test_correct_intent"+str(epoch)+".pkl", "wb" ) )
    pickle.dump(test_pred_slot, open( "./results2/test_pred_slot"+str(epoch)+".pkl", "wb" ) )
    pickle.dump(test_correct_slot, open( "./results2/test_correct_slot"+str(epoch)+".pkl", "wb" ) )
    if (epoch+1)%10 == 1:
      torch.save(intent_model, 'model_intent'+str(epoch)+'.ckpt')
      torch.save(slot_model, 'model_slot'+str(epoch)+'.ckpt')
    score = accuracy_score(test_correct_intent, test_pred_intent)
    F1_score = 0
    # F1_score = 100.0*2*TP/(2*TP+FN+FP)
    # if F1_score > best_F1_score:
    #     best_F1_score = F1_score
    #     best_epoch_slot = epoch
    print('*'*20)
    print('Epoch: [{}/{}], Intent Val Acc: {:.4f} \t Slot F1 score: {:.4f}'.format(epoch+1, epoch_num, score, F1_score))
    print('*'*20)
    
    # print('Best Intent Acc: {:.4f} at Epoch: [{}]'.format(100.0*best_correct_num/total_test, best_epoch+1))
    # print('Best F1 score: {:.4f} at Epoch: [{}]'.format(best_F1_score, best_epoch_slot+1))

