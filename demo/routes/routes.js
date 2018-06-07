module.exports = function (express) {
    var router = express.Router();
    var api_root = '/api';
    router.get(api_root, function (req, res) {
        res.json({ 'success' : true, 'message' : 'Welcome', 'status' : 'Online' });
    });
    return router;
};
