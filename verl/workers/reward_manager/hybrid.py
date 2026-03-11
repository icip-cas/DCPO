from collections import defaultdict
from typing import Any

import torch
import re

from verl import DataProto
from verl.utils.reward_score import _default_compute_score

def reward_func(prompt, response):

    conf = None
    match = re.search(r"CONFIDENCE:\s*([01]?\.\d+|\d+)", response)

    if match:
        pos = match.start()
        try:
            conf = float(match.group(1))
        except:
            conf = 0.0

    # 若模型未给出，则设默认值
    if conf is None:
        conf = 0.0

    return conf

def token_reward_func(prompt, response, tokenizer):
    conf = None
    conf_token_pos = None

    match = re.search(r"CONFIDENCE:\s*([01]?\.\d+|\d+)", response)

    if match:
        try:
            conf = float(match.group(1))
            char_pos = match.start()
        except:
            conf = -1

        # ===== 核心：字符位置 -> token 位置 =====
        # CONFIDENCE 起始字符位置（在 response 内）
        char_pos = match.start()

        # 将 response 按 tokenizer 编码
        response_ids = tokenizer.encode(response, add_special_tokens=False)

        # 逐 token 解码，累计字符长度，定位 token index
        cum_len = 0
        for idx, tok_id in enumerate(response_ids):
            tok_str = tokenizer.decode([tok_id])
            cum_len += len(tok_str)
            if cum_len >= char_pos:
                conf_token_pos = idx
                break
    else:
        conf = -1
        conf_token_pos = None

    return conf, conf_token_pos

class HybridRewardManager:
    def __init__(self, tokenizer, num_examine, compute_score = None, reward_fn_key="data_source", seperate_confidence = True)->None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.seperate_confidence = seperate_confidence

    def __call__(self, data: DataProto, return_dict: bool = False, save_analysis_path=None) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        seperate_confidence = self.seperate_confidence

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        
        # data.batch keys:
        # 1. responses: response tokens
        # 2. prompts: 
        confidence_score_list = []
        accuracy_score_list = []
        confidence_pos_list = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            
            # custom score
            prompt = self.tokenizer.decode(valid_prompt_ids)
            response = self.tokenizer.decode(valid_response_ids)
            # 这里可以写你自定义的reward函数

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores

            score = self.compute_score(
                data_source=data_source,
                solution_str=response,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            if seperate_confidence:
                confidence_score, confidence_pos = token_reward_func(prompt=prompt, response=response, tokenizer=self.tokenizer)
                
                if confidence_pos is not None:
                    pos = max(confidence_pos - 1, 0)
                    reward_tensor[i, pos] = reward
                    confidence_pos_list.append(pos)
                    extra_info["confidence_pos"] = confidence_pos
                else:
                    pos = max(valid_response_length - 2, 0)
                    reward_tensor[i, pos] = reward
                    confidence_pos_list.append(pos)
                    extra_info["confidence_pos"] = valid_response_length - 1
            else:
                confidence_score = reward_func(prompt=prompt, response=response)
                reward_tensor[i, valid_response_length - 1] = reward
                confidence_pos_list.append(valid_response_length - 1)

            confidence_score_list.append(confidence_score)
            accuracy_score_list.append(reward)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt)
                print("[response]", response)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        accuracy_tensor = torch.tensor(accuracy_score_list)
        confidence_tensor = torch.tensor(confidence_score_list)
        pos = confidence_pos_list

        # calibration: mean accuracy - individual confidence
        num_rollouts = 8

        if num_rollouts is None:
            # fallback：从 batch size 和 prompt 数推断
            # 官方 GRPO 通常 batch_size = num_prompts * num_rollouts
            raise ValueError("num_rollouts must be provided for GRPO")

        batch_size = len(accuracy_score_list)
        if batch_size % num_rollouts != 0:
            num_rollouts = 1  # cannot group, treat each separately

        num_groups = batch_size // num_rollouts

        # (num_groups, num_rollouts)
        accuracy_grouped = accuracy_tensor.view(num_groups, num_rollouts)

        # group mean accuracy: (num_groups,)
        group_mean_accuracy = accuracy_grouped.mean(dim=1)

        # broadcast back to rollout level: (batch_size,)
        group_mean_accuracy_expanded = group_mean_accuracy.repeat_interleave(num_rollouts)

        gamma = 0.5

        hybrid_accuracy_tensor = gamma * accuracy_tensor + (1 - gamma) * group_mean_accuracy_expanded

        calibration_scores = torch.square(
           hybrid_accuracy_tensor - confidence_tensor
        )

        weight = 0.5

        for i in range(len(data)):
        # get the position used previously
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

            reward_tensor[i, valid_response_length - 1] -= weight * calibration_scores[i]

        reward_extra_info["confidence_pos"] = pos
        reward_extra_info["confidence"] = confidence_score_list
        reward_extra_info['accuracy'] = accuracy_score_list
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }

        return reward_tensor