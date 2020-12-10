#!/usr/bin/python3
# coding: utf-8

# In[9]:


import tensorflow as tf

import tensorflow_recommenders as tfrs

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime, date, timedelta


cupom = pd.read_csv("./dados/cupons.csv")
cupom['id_cliente']  = cupom['id_cliente'].astype('str')
material = pd.read_csv("./dados/material.csv")

# In[89]:


tf_material = (
    tf.data.Dataset.from_tensor_slices({
        "nome_material" : tf.cast(material['nome_material'].values, tf.string)
    })
)


# In[90]:


tf_cupom = (
    tf.data.Dataset.from_tensor_slices(
        {
            "id_cliente" : tf.cast(cupom['id_cliente'].values, tf.string),
            "nome_material" : tf.cast(cupom['nome_material'].values, tf.string)
        }
    )
)


# In[91]:


tf_cupom = tf_cupom.map(lambda x: {
    "id_cliente": x["id_cliente"],
    "nome_material": x["nome_material"],
})

tf_material = tf_material.map(lambda x: x["nome_material"])


material.nome_material.nunique()


# In[108]:


cupom.shape


# In[99]:


cupom.id_cliente.nunique()


# In[109]:


class CuponsModel(tfrs.Model):
  """Client prediction model."""

  def __init__(self):
    # The `__init__` method sets up the model architecture.
    super().__init__()

    # How large the representation vectors are for inputs: larger vectors make
    # for a more expressive model but may cause over-fitting.
    embedding_dim = 32
    num_unique_users = 9953
    num_unique_movies = 66838
    eval_batch_size = 128

    self.user_model = tf.keras.Sequential([
      # We first turn the raw user ids into contiguous integers by looking them
      # up in a vocabulary.
      tf.keras.layers.experimental.preprocessing.StringLookup(
          max_tokens=num_unique_users),
      # We then map the result into embedding vectors.
      tf.keras.layers.Embedding(num_unique_users, embedding_dim)
    ])

    self.material_model = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.StringLookup(
          max_tokens=num_unique_movies),
      tf.keras.layers.Embedding(num_unique_movies, embedding_dim)
    ])

# The `Task` objects has two purposes: (1) it computes the loss and (2)
    # keeps track of metrics.
    self.task = tfrs.tasks.Retrieval(
        # In this case, our metrics are top-k metrics: given a user and a known
        # watched movie, how highly would the model rank the true movie out of
        # all possible movies?
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=tf_material.batch(eval_batch_size).map(self.material_model)
        )
    )
  def compute_loss(self, features, training=False):
    # The `compute_loss` method determines how loss is computed.

    # Compute user and item embeddings.
    user_embeddings = self.user_model(features["id_cliente"])
    material_embeddings = self.material_model(features["nome_material"])

    # Pass them into the task to get the resulting loss. The lower the loss is, the
    # better the model is at telling apart true watches from watches that did
    # not happen in the training data.
    return self.task(user_embeddings, material_embeddings)


# In[110]:


model = CuponsModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

model.fit(tf_cupom.batch(10000), verbose=False)


# In[112]:


index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
index.index(tf_material.batch(1000).map(model.material_model), tf_material)

# Get recommendations.
_, titles = index(tf.constant(["46355"]))
print("Recommendations for user 46355: {titles[0, :3]}")


# In[116]:


_, titles = index(tf.constant(["194466"]))
print("Recommendations for user 194466: {titles[0, :3]}")


# In[128]:


index(tf.constant(["3265"]))


# In[97]:


type(index)


# In[115]:


cupom['id_cliente'].unique()[0:5]


# In[103]:


import tempfile
import os


# In[107]:


# # Export the query model.
# with tempfile.TemporaryDirectory() as tmp:
#   path = os.path.join(tmp, "model")

# Save the index.
index.save("model")

# Load it back; can also be done in TensorFlow Serving.
loaded = tf.keras.models.load_model("model")

# Pass a user id in, get top predicted movie titles back.
scores, titles = loaded(["28573"])

print("Recommendations: {titles[0][:3]}")
