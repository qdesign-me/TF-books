import data from './data/books.json';
import { Card, Rate, Row, Col, Form, Input, Button } from 'antd';
import { useEffect, useMemo, useState } from 'react';
import * as tf from '@tensorflow/tfjs';

import 'antd/dist/antd.css';
import './styles/style.css';

const book_arr = tf.range(0, data.length);
const book_len = data.length;

function App() {
  const [user, setUser] = useState(null);
  const [books, setBooks] = useState([]);
  const [ready, setReady] = useState(false);
  const [loading, setLoading] = useState(false);
  const [model, setModel] = useState(null);

  useEffect(() => {
    const loadModel = async () => {
      console.log('Loading Model...');
      const model = await tf.loadLayersModel('http://localhost:3000/model/model.json', false);
      console.log('Model Loaded Successfull');
      setModel(model);
      setReady(true);
    };
    loadModel();
  }, []);
  // useEffect(() => {
  //   const loadModel = async () => {
  //     console.log('Loading Model...');
  //     const csvUrl = 'http://localhost:3000/model/ratings.csv';
  //     const trainingData = tf.data.csv(csvUrl, {
  //       hasHeader: true,
  //       columnNames: ['book_id', user_id', 'rating'],
  //       columnConfigs: {
  //         user_id: {
  //           isLabel: true,
  //         },
  //       },
  //     });

  //     const users = [];
  //     const books = [];
  //     await trainingData.forEachAsync((e) => {
  //       const userId = e.ys.user_id;
  //       const bookId = e.xs.book_id;
  //       if (!users.includes(userId)) users.push(userId);
  //       if (!books.includes(bookId)) books.push(bookId);
  //     });
  //     console.log(users);
  //     console.log(books);

  //     const input_books = tf.layers.input({ shape: [1] });
  //     const embed_books = tf.layers.embedding({ outputDim: 15, inputDim: 1 }).apply(input_books);
  //     const books_out = tf.layers.flatten().apply(embed_books);

  //     const input_users = tf.layers.input({ shape: [1] });
  //     const embed_users = tf.layers.embedding({ outputDim: 15, inputDim: 1 }).apply(input_users);
  //     const users_out = tf.layers.flatten().apply(embed_users);

  //     const conc_layer = tf.layers.concatenate().apply([books_out, users_out]);
  //     const x = tf.layers.dense({ units: 128, activation: 'relu' }).apply(conc_layer);
  //     const y = tf.layers.dense({ units: 1, activation: 'relu' }).apply(x);
  //     const model = tf.model({ inputs: [input_books, input_users], outputs: y });
  //     model.compile({ loss: 'meanSquaredError', optimizer: 'adam' });

  //     model.summary();
  //   };
  //   loadModel();
  // }, []);

  const recommend = async (userId) => {
    setLoading(true);
    let user = tf.fill([book_len], Number(userId));

    console.log(`Recommending for User: ${userId}`);
    let pred_tensor = await model.predict([book_arr, user]).reshape([10000]);
    const pred = pred_tensor.arraySync();

    let recommendations = [];
    for (let i = 0; i < 6; i++) {
      const max = pred_tensor.argMax().arraySync();

      recommendations.push(data[max]); //Push book with highest prediction probability
      pred.splice(max, 1); //drop from array

      pred_tensor = tf.tensor(pred); //create a new tensor
    }
    setLoading(false);
    return recommendations;
  };

  useEffect(() => {
    if (!user) {
      // no info about user, return top 6 books
      return setBooks(data.sort((a, b) => b.average_rating - a.average_rating).slice(0, 6));
    }
    recommend(user).then((data) => {
      setBooks(data);
    });
  }, [user]);
  const onLogin = (data) => {
    setUser(+data.user);
  };

  return (
    <div className="container">
      <Row gutter={[24, 24]}>
        {books?.map((book, index) => (
          <Col span={4} key={index}>
            <Card hoverable cover={<img alt="example" src={book.image_url} />}>
              <Card.Meta title={book.title} description={<Rate disabled allowHalf size="small" defaultValue={book.average_rating} />} />
            </Card>
          </Col>
        ))}
      </Row>
      --{JSON.stringify(loading)}---
      {!ready && <p>TF model is loading....</p>}
      {loading && <p>loading predictions....</p>}
      {ready && !loading && (
        <Form style={{ marginTop: 24 }} layout="inline" onFinish={onLogin}>
          <Form.Item label="Login as user" name="user">
            <Input />
          </Form.Item>
          <Button type="primary" htmlType="submit">
            Submit
          </Button>
        </Form>
      )}
    </div>
  );
}

export default App;
