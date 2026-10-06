

def process_all_threads(input_folder, output_folder, if_write="no"):
	if if_write == "yes":
		os.makedirs(output_folder, exist_ok=True)
	elif if_write == "no":
		total_conversations = []
		conversation_labels = [] #label for the whole conversation
	else:
		raise ValueError("if_write must be either 'yes' or 'no'")
	# List all CSV files in the folder
	csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]
	for csv_file in csv_files:
		csv_path = os.path.join(input_folder, csv_file)
		df = pd.read_csv(csv_path)
		df = df[df["aggressive"].notna() & (df["aggressive"] != "")] #filter out comments without aggressive label
		# binarize AG labels
		df["aggressive"] = df["aggressive"].replace({"OAG": "AG", "CAG": "AG"})
		# Build parent → children mapping
		children = {}
		for _, row in df.iterrows():
			parent = row["Reply_To"]
			msg_id = row["Message_ID"]
			children.setdefault(parent, []).append(msg_id) #each parent has a list, appended with the child comment id
		#Identify root messages
		all_ids = set(df["Message_ID"])
		#Selects messages where Reply_To is empty / NaN, and messages whose Reply_To value is not in the dataset.
		root_ids = df[df["Reply_To"].isna() | ~df["Reply_To"].isin(all_ids)]["Message_ID"].tolist() 
		# DFS to build branches
		branches = []
		def dfs(current_id, path):
			path.append(current_id)
			if current_id not in children:
				branches.append(path.copy())
			else:
				for child in children[current_id]:
					dfs(child, path)
			path.pop()

		for r in root_ids:
			dfs(r, [])

		# Save each branch as a separate CSV
		base_name = os.path.splitext(csv_file)[0]
		for i, branch in enumerate(branches, start=1):
			if len(branch) < 3:                                    #skip branches with less than 3 messages
				continue
			##These lines are used to check the labels of the first two messages in a branch, for example, to discard branches where the first two comments are both aggressive.
			first_label = df[df["Message_ID"] == branch[0]]["aggressive"].values[0]
			second_label = df[df["Message_ID"] == branch[1]]["aggressive"].values[0]
			if first_label == "AG" or second_label == "AG":         #the first exchange should not be uncivil
				continue
			branch_df = df[df["Message_ID"].isin(branch)].copy()
			##branch_df = branch_df.sort_values(by="time"). ##better keep the parent child order?  ##from 
			ag_indices = branch_df.index[branch_df["aggressive"] == "AG"].tolist()# if a conversation has AG label, label it as AG; otherwise civil throughout
			if len(ag_indices) > 0:
				first_ag_index = ag_indices[0]
				branch_df = branch_df.loc[:first_ag_index]    #only keep conversation up to and including the first AG message
				branch_label = "AG"
			else:
				branch_label = "NAG"
			
			#branch_df = assign_author_based_turns(branch_df) 
			branch_df["Turn_ID"] = range(1, len(branch_df) + 1)

			if if_write == "yes":
				out_file = f"{base_name}_{i}_{branch_label}.csv"
				out_path = os.path.join(output_folder, out_file)
				branch_df.to_csv(out_path, index=False)
			elif if_write == "no":
				total_conversations.append(branch_df)
				conversation_labels.append(branch_label)
			else:
				raise ValueError("if_write must be either 'yes' or 'no'")
			#label for the whole conversation is shown in the file name and the last "aggressive" label
	if if_write == "no":
		return total_conversations, conversation_labels
	else:
		return None, None		

if __name__ == "__main__":
	input_folder = " " #human annotation folder
	output_folder = "" 
	conversation_ls, conversation_lbs = process_all_threads(input_folder, if_write="no")
	print(f"Total conversations processed: {len(conversation_ls)}")
