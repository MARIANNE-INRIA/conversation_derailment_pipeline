
def parse_annotations(cell, is_gpt=False):
	print("type of input:", type(cell))
	try:
		if isinstance(cell, str):
			if cell.strip().lower() in ["literal", "none"]:  
				return "literal"  
			if not is_gpt:
				cell = "[" + cell.strip().strip(",") + "]" 
		ann = json.loads(cell)
		if isinstance(ann, dict) and is_gpt: #json format, gpt annotation output
			return [
				{"content": v["content"], "type": v["type"], "confidence": v.get("confidence", 1.0)}
				for v in ann.values()
			]
		elif isinstance(ann, list):
			return ann
	except Exception as e:
		print("Parse error:", e, "Cell:", cell)
		return []
	return []


def most_salient_inference_extract(pragmatic_inf_typ, most_salient_inference, gpt_or_human="gpt"):
	if gpt_or_human.lower() == "gpt": 	
		try:
			salient_index = most_salient_inference   
			most_salient_inference_text = pragmatic_inf_typ[salient_index]["content"]
		except Exception:
			most_salient_inference_text = ""
	elif gpt_or_human.lower() == "human":
		most_salient_inference_text = str(most_salient_inference)
	else:
		raise ValueError("gpt_or_human must be either 'gpt' or 'human'")
	return most_salient_inference_text


def generate_inference_based_template(conversation_ls, conversation_lbs, gpt_or_human="gpt", focus="message"):
	generated_template_total = []
	total_conv_lbs = []
	total_conv_ids = []
	total_turn_ids = []
	#assert len(conversation_ls) == len(conversation_lbs)
	for conv, lb in zip(conversation_ls, conversation_lbs):	
		generated_template_conv = []
		conv_turn_ids = []
		conv_ids = []
		conv_lbs = []
		cumulative_templates = []    
		
		for idx, (_, row) in enumerate(conv.iterrows(), start=1):
			prag_inf = row["Pragmatic_Inferences"].strip()
			print("prag_inf raw:", prag_inf)
			author = str(row["Message_Author"])
			conv_id = str(row["Subreddit"])
			turn_id = row["Turn_ID"]
			content = str(row["Message_Content"])
			most_salient_inference = row["most_salient_inference"]
			inf_typ = row["inference_type"]
			as_intended = row["as_intended"]
			PRE_IMP = row["PRE/IMP"] 
			
			pragmatic_inf_typ_ls = parse_annotations(prag_inf, is_gpt=(gpt_or_human=="gpt")) 
			most_salient_inference_text = most_salient_inference_extract(prag_inf, most_salient_inference, gpt_or_human= gpt_or_human)
			
			if pragmatic_inf_typ_ls == "literal":
				pragmatic_inf_content_txt = (f"The comment does not contain any pragmatic inferences; it is literal."
				f"The speaker said: {content} ")
			elif len(pragmatic_inf_typ_ls) == 0:
				pragmatic_inf_content_txt = (f"The comment does not contain any pragmatic inferences."
				f"The speaker said: {content} ")
			elif pragmatic_inf_typ_ls != "literal" and len(pragmatic_inf_typ_ls) == 1:
				pragmatic_inf = pragmatic_inf_typ_ls[0]
				pragmatic_inf_typ = pragmatic_inf.get("type", "")
				pragmatic_inf_content = pragmatic_inf.get("content", "") 
				pragmatic_inf_content_txt = (f"A pragmatic inference can be made. It is of {pragmatic_inf_typ} speech act type."
				f"The inferred meaning is: {pragmatic_inf_content} ")
			elif pragmatic_inf_typ_ls != "literal" and len(pragmatic_inf_typ_ls) > 1:
				print("multiple pragmatic inferences!:", len(pragmatic_inf_typ_ls))
				inference_texts = []
				pragmatic_inf = pragmatic_inf_typ_ls[0]
				pragmatic_inf_typ = pragmatic_inf.get("type", "")
				pragmatic_inf_content = pragmatic_inf.get("content", "")
				pragmatic_inf_content_txt_1 = (f"A pragmatic inference can be made. It is of {pragmatic_inf_typ} speech act type."
				f"The inferred meaning is: {pragmatic_inf_content} ")
				inference_texts.append(pragmatic_inf_content_txt_1)
				for i, pragmatic_inf in enumerate(pragmatic_inf_typ_ls[1:]):
					pragmatic_inf_typ = pragmatic_inf.get("type", "")
					pragmatic_inf_content = pragmatic_inf.get("content", "")
					txt = (
					f"Another pragmatic inference can be made. It is of {pragmatic_inf_typ} speech act type. "
					f"The inferred meaning is: {pragmatic_inf_content} "
					)
					inference_texts.append(txt)
				pragmatic_inf_content_txt = " ".join(inference_texts)
			else:
				pragmatic_inf_content_txt = ''
			most_salient_inf_txt = (f"For the conversation flow, the most salient pragmatic inference is {most_salient_inference_text} "
						   f"The predominent inference type is {inf_typ} speech act type.")
			if as_intended.lower() == "yes":    
				as_intended_inf_txt = "The speaker's most salient message is agreed by the listener."
			elif as_intended.lower() == "no":
				as_intended_inf_txt = "The speaker's most salient message is not agreed by the listener."
			else:
				as_intended_inf_txt = "It is uncertain whether the speaker's most salient message is agreed by the listener."
			#the most salient inference is the speaker's {information, emotion, suggestion, determination, declarative action} that {most salient inference text}
			#the reply message does not agree with the speaker's {information, attitude, determination, declarative action}--depending on the inference type of the most salient inference 
			#the most salient inference is {a presupposition/an implicature}
			#the overall tone is {aggressive/non-aggressive}
			if PRE_IMP is None or (isinstance(PRE_IMP, float) and np.isnan(PRE_IMP)):
				PRE_IMP_txt = "The presupposition/implicature distinction is not applicable. "
			elif PRE_IMP.lower() == "pre":
				PRE_IMP_txt = "The most salient inference is a presupposition. "
			elif PRE_IMP.lower() == "imp":
				PRE_IMP_txt = "The most salient inference is an implicature. "
			else:
				PRE_IMP_txt = "It is unclear whether the most salient inference is a presupposition or an implicature. "
			msg_template = (f"This is message {idx}. "
						f"The message is authored by {author}. "
						f"The content of the message is: {content} ")
			prag_template = (
				f"This is message {idx}. "
				f"The message is authored by {author}. "
				f"{pragmatic_inf_content_txt} "
				f"{most_salient_inf_txt} "
				f"{PRE_IMP_txt} "
				f"{as_intended_inf_txt} ")
			
			msg_prag_full_template = (
				f"This is message {idx}. "
				f"The message is authored by {author}. "
				f"The content of the message is: {content} "
				f"{pragmatic_inf_content_txt} "
				f"{most_salient_inf_txt} "
				f"{PRE_IMP_txt} "
				f"{as_intended_inf_txt} ")
			
			most_salient_template = (
				f"This is message {idx}. "
				f"The message is authored by {author}. "
				f"{most_salient_inference_text} "
				f"{PRE_IMP_txt} "
				f"{as_intended_inf_txt} ")
			
			msg_prag_salient_template = (
				f"This is message {idx}. "
				f"The message is authored by {author}. "
				f"The content of the message is: {content} "
				f"{pragmatic_inf_content_txt} "
				f"{most_salient_inf_txt} "
			)
			
			msg_salient_template = (
				f"This is message {idx}. "
				f"The message is authored by {author}. "
				f"The content of the message is: {content} "
				f"{most_salient_inf_txt} "
			)

			msg_prag_template = (
				f"This is message {idx}. "
				f"The message is authored by {author}. "
				f"The content of the message is: {content} "
				f"{pragmatic_inf_content_txt} "
			)
		
			if focus.lower() =="message":
				current_template = msg_template
			elif focus.lower() == "pragmatics":
				current_template = prag_template
			elif focus.lower() == "salient":
				current_template = most_salient_template
			elif focus.lower() == "msg_prag_salient":
				current_template = msg_prag_salient_template
			elif focus.lower() == "msg_salient":
				current_template = msg_salient_template
			elif focus.lower() == "msg_prag":
				current_template = msg_prag_template
			elif focus.lower() == "msg_prag_full":
				current_template = msg_prag_full_template
			else:
				raise ValueError("focus must be either 'message', 'pragmatics', 'salient', 'msg_salient', 'msg_prag', 'msg_prag_salient' or 'msg_prag_full' ")

			# ---- NEW PART: build cumulative prompt ----
			cumulative_templates.append(current_template)
			# Combine all previous templates into one prompt
			# cumulative_prompt = "\n".join(cumulative_templates)
			cumulative_prompt = "\n".join(
    			[f"[Previous message {i}]\n{t}" for i, t in enumerate(cumulative_templates[:-1], start=1)] +
    				[f"[Current message]\n{current_template}"])

			# Store the rolling prompt
			generated_template_conv.append(cumulative_prompt) #can be used to generate summaries accordingly;
	
			conv_turn_ids.append(turn_id)
			conv_ids.append(conv_id)
			conv_lbs.append(lb)
		total_conv_lbs.append(conv_lbs)        #each conv has one label 
		total_conv_ids.append(conv_ids)     #each conv has a list of ids
		total_turn_ids.append(conv_turn_ids)    #each conv has a list of turn ids
		generated_template_total.append(generated_template_conv) #each conv has a list of strings/comment 
	return generated_template_total, total_conv_lbs, total_conv_ids, total_turn_ids






