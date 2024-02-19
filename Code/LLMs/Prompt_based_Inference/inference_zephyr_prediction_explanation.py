# %%capture
# !pip install git+https://github.com/huggingface/transformers.git
# !pip install accelerate

import torch
from transformers import pipeline
from tqdm import tqdm
#torch.set_default_device('cuda')
pipe = pipeline("text-generation",
                model="HuggingFaceH4/zephyr-7b-beta",
                torch_dtype=torch.bfloat16,
                device_map="auto")
import pandas as pd
df_test = pd.read_csv("test.csv")
def preprocess_case(text):
    max_tokens = 2100
    tokens = text.split()
    num_tokens_to_extract = min(max_tokens, len(tokens))
    text1 = ' '.join(tokens[-num_tokens_to_extract:])
    return text1

df_test["zephyr-7b-beta"] = ""

for i,row in tqdm(df_test.iterrows()):
    case_text = preprocess_case(row['Input'])

    template = f"""Task: Given a Supreme Court of India case proceeding enclosed in angle brackets <>, your task is to predict the decision/verdict of the case (with respect to the appellant). \

    Prediction: Given a case proceeding, the task is to predict the decision 0 or 1, where the label 1 corresponds to the acceptance of the appeal/petition \
    of the appellant/petitioner and the label 0 corresponds to the rejection of the appeal/petition of the appellant/petitioner. \

    Context: Answer in a consistent style as shown in the following two examples: \

    case_proceeding: Leave granted. In 2008, the Punjab State Water Supply Sewerage Board, Bhatinda issued numberice inviting tender for extension and augmentation of water supply, sewerage scheme, pumping station and sewerage treatment plant for various towns mentioned therein on a turnkey basis. On 25.9.2008, the appellant companypany, which is Signature Not Verified involved in civil electrical works in India, was awarded the said Digitally signed by NIDHI AHUJA Date 2019.03.11 173359 IST Reason tender after having been found to be the best suited for the task. On 16.1.2009, a formal companytract was entered into between the appellant and respondent No. 2. It may be mentioned that the numberice inviting tender formed part and parcel of the formal agreement. \
    What happened in the present case is certainly a paradoxical situation which should be avoided. Total companytract is for Rs. 12,23,500. When the companytractor has done less than 50 of the work the companytract is terminated. He has been paid Rs 5,71,900. In a Section 20 petition he makes a claim of Rs. 39,47,000 and before the arbitrator the claim is inflated to Rs. 63,61,000. He gets away with Rs. 20,08,000 with interest at the rate of 10 per annum and penal interest at the rate of 18 per annum. Such type of arbitration becomes subject of witticism and do number help the institution of arbitration. Rather it brings a bad name to the arbitration process as a whole. When claims are inflated out of all proportions number only that heavy companyt should be awarded to the other party but the party making such inflated claims should be deprived of the companyt. We, therefore, set aside the award of companyt of Rs. 7500 given in favour of the companytractor and against the State of Jammu and Kashmir.  Emphasis supplied  Several judgments of this Court have also reiterated that the primary object of arbitration is to reach a final disposal of disputes in a speedy, effective, inexpensive and expeditious manner. Thus, in Centrotrade Minerals Metal Inc. v. Hindustan Copper Ltd.,  2017  2 SCC 228, this companyrt held In Union of India v. U.P. State Bridge Corpn. Ltd.  2015  2 SCC 52 this Court accepted the view O.P. Malhotra on the Law and Practice of Arbitration and Conciliation  3rd Edn. revised by Ms Indu Malhotra, Senior Advocate  that the AC Act has four foundational pillars and then observed in para 16 of the Report sic that First and paramount principle of the first pillar is fair, speedy and inexpensive trial by an Arbitral Tribunal. Unnecessary delay or expense would frustrate the very purpose of arbitration. Interestingly, the second principle which is recognised in the Act is the party autonomy in the choice of procedure. This means that if a particular procedure is prescribed in the arbitration agreement which the parties have agreed to, that has to be generally resorted to.  Emphasis in original  Similarly, in Union of India v. Varindera Constructions Ltd.,  2018  7 SCC 794, this Court held- The primary object of the arbitration is to reach a final disposition in a speedy, effective, inexpensive and expeditious manner. In order to regulate the law regarding arbitration, legislature came up with legislation which is known as Arbitration and Conciliation Act, 1996. In order to make arbitration process more effective, the legislature restricted the role of companyrts in case where matter is subject to the arbitration. Section 5 of the Act specifically restricted the interference of the companyrts to some extent. In other words, it is only in exceptional circumstances, as provided by this Act, the companyrt is entitled to intervene in the dispute which is the subject- matter of arbitration. Such intervention may be before, at or after the arbitration proceeding, as the case may be. \

    Prediction: 1 \

    case_proceeding: SANTOSH HEGDE, J. Noticing certain companytradictory views in three different judgments of this Court in Teg Singh vs. Charan Singh  1977  2 SCC 732, Kesar Singh vs. Sadhu  1996  7 SCC 711  and Balwant Singh vs. Daulat Singh  1997  7 SCC 137 , a Division Bench of 2-Judges of this Court referred the instant appeals for disposal by a larger bench by its referral order dated 27th October, 2004, hence, this appeal is before us. Brief facts giving rise to these appeals are as follows One Hirday Ram was the owner of the suit property. He had three wives, namely, Kubja, Pari and Uttamdassi. Kubja had pre- deceased Hirday Ram leaving behind a daughter named Tikami. During his life time, Hirday Ram made a Will dated 1.10.1938 whereby he bequeathed a part of his property to his daughter Tikami and the remaining property was given to his two other wives, named above, for their maintenance with the companydition that they would number have the power to alienate the same in any manner. As per the Will, after the death of the above two wives of Hirday Ram, the property was to revert back to his daughter Tikami as absolute owner. \
    A jurisdictional question if wrongly decided would number attract the principle of res judicata. When an order is passed without jurisdiction, the same becomes a nullity. When an order is a nullity, it cannot be supported by invoking the procedural principles like, estoppel, waiver or res judicata. It would, therefore, be number companyrect to companytend that the decision of the learned Single Judge attained finality and, thus, the principle of res judicata shall be attracted in the instant case. From the above principles laid down by this Court, it is clear that if the earlier judgment which is sought to be made the basis of res judicata is delivered by a companyrt without jurisdiction or is companytrary to the existing law at the time the issue companyes up for reconsideration such earlier judgment cannot be held to be res judicata in the subsequent case unless, of companyrse, protected by any special enactment. \
    It is true that the judgment in Tulasammas case is number retrospective and would number apply to cases which have ended finally. But a declaratory decree simplicitor does number attain finality if it has to be used for obtaining any future decree like possession. In such cases of suit for possession based on an earlier declaratory decree is filed it is open to the defendant to establish that the declaratory decree on which suit is based is number a lawful decree. Unfortunately for the appellant the declaration obtained by her based on which she was seeking possession in the present suit being companytrary to law, the companyrts below companyrectly held that the appellant companyld number seek possession on the basis of such an illegal declaration. Thus, the law is clear on this point i.e. if a suit is based on an earlier decree and such decree is companytrary to the law prevailing at the time of its companysideration as to its legality or is a decree granted by a companyrt which has numberjurisdiction to grant such decree, principles of res judicata under Section 11 of the CPC will number be attracted and it is open to the defendant in such suits to establish that the decree relied upon by the plaintiff is number a good law or companyrt granting such decree did number have the jurisdiction to grant such decree. In the instant case, as numbericed hereinabove, the present suit is filed for possession of the suit properties on the basis of a declaratory decree obtained earlier which is found to be number a lawful decree as per the law prevailing at present. Hence, the impugned judgment cannot be interfered with. Thus, examined from any angle, we do number find any merit in this appeal. \

    Prediction: 0 \

    Instructions: Learn from the above given two examples and perform the task for the following case proceeding. \

    case_proceeding: <{case_text}>

    Format your output in list format: [prediction, explanation] """

    messages = [
        {
            "role": "system",
            "content": "You are an honest legal advisor.",
        },
        {"role": "user", "content": template},
    ]

    prompt = pipe.tokenizer.apply_chat_template(messages,
                                                tokenize=False,
                                                add_generation_prompt=True)
    outputs = pipe(prompt,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.1,
                top_k=50,
                top_p=0.95,
                pad_token_id=50256)
    response = outputs[0]["generated_text"]
    verdict = response.split("<|assistant|>\n")[1]
    df_test.at[i, 'zephyr-7b-beta'] = verdict
#     print("Actual Outcome: ", row["Label"])
#     print(verdict[:30])
#     print("==========================================================")
#     if i==20:
#         break
df_test.to_csv('results/zephyr_pred_exp.csv', index=False)
